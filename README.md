# vLLM на CPU в Kubernetes

Инференс Qwen2.5-1.5B-Instruct (квантизация W8A8 int8,
[`RedHatAI/Qwen2.5-1.5B-Instruct-quantized.w8a8`](https://huggingface.co/RedHatAI/Qwen2.5-1.5B-Instruct-quantized.w8a8))
через vLLM 0.32.0 (CPU-сборка) в minikube.
OpenAI-совместимый API закрыт Bearer-токеном, контейнеры работают без root,
веса кэшируются на PVC, промпты и ответы в логи не попадают.

```
.
├── docker/
│   ├── vllm/                  # основной образ: Dockerfile, requirements.in/.txt, entrypoint.sh
│   └── model-downloader/      # лёгкий образ для initContainer: Dockerfile, download_model.py
├── privacy_logging/           # фильтр логов + logging_config.json для vLLM
├── k8s/
│   ├── base/                  # kustomize (бонус 1): namespace, pvc, deployment, service, генераторы ConfigMap/Secret
│   └── bench/                 # Job с нагрузочным тестом (vllm bench serve)
├── tests/                     # pytest: фильтр логов и загрузчик модели
├── .github/workflows/ci.yml   # тесты, hadolint, shellcheck, kustomize + kubeconform
├── bench/results/             # результаты прогона
└── Makefile
```

## Быстрый старт

Нужны: docker, minikube, kubectl (kustomize встроен). Проверялось на
minikube v1.39 (driver docker, 12 CPU, 12 ГБ RAM), Intel с AVX2 и AVX-VNNI.

```bash
minikube start --driver=docker --cpus=12 --memory=12g

# 1. Образы. Собираются на хосте и загружаются в minikube.
make images        # docker build для обоих Dockerfile (без buildx — классическим сборщиком)
make load          # minikube image load

# 2. Секреты. Создаёт k8s/base/secrets.env со случайным VLLM_API_KEY.
#    Для gated-модели перед этим: export HF_TOKEN=hf_...
make k8s/base/secrets.env

# 3. Деплой
make deploy        # kubectl apply -k k8s/base + ожидание rollout
```

Первый запуск долгий: initContainer качает ~2 ГБ весов на PVC, потом
vLLM компилирует граф (ещё 5–10 минут на CPU). Кэш компиляции тоже лежит на PVC, поэтому следующие рестарты пода
занимают 2–4 минуты и в сеть не ходят: в логе initContainer'а будет
`Model ... is already in /models/..., skipping download`.

Важно: minikube должен получить памяти не меньше 12 ГБ. Если профиль
уже был создан раньше с меньшим объёмом, `--memory` при `minikube start`
не применится; для driver=docker можно поднять лимит на лету:
`docker update --memory 12g --memory-swap 12g minikube`.

### Запрос к модели

```bash
kubectl -n llm port-forward svc/vllm 8000:8000 &
TOKEN=$(make -s token)

curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct-w8a8",
       "messages": [{"role": "user", "content": "Столица Франции?"}],
       "max_tokens": 32}'
```

Без токена или с неверным токеном сервер отвечает `401`.

## Как устроено

### Образ vLLM (`docker/vllm`)

* База `python:3.12-slim-trixie`, закреплена по digest. CPU-колесо vLLM 0.32.0
  собрано под glibc 2.39, в trixie 2.41, в bookworm уже не подходит.
* Двухстадийная сборка: в builder ставится venv, в финальный образ копируется
  только он. Компиляторы pip не нужны, но в рантайме оставлен `g++`:
  torch.compile на CPU генерирует C++ ядра при старте.
* Пользователь `vllm` (uid 10001), `USER` задан числом, чтобы
  `runAsNonRoot` в k8s мог это проверить.
* `LD_PRELOAD` с tcmalloc и Intel OpenMP, как рекомендует документация vLLM для CPU.
* `TRITON_CACHE_DIR` и кэш torch.compile направлены в `~/.cache`, который в
  поде смонтирован с PVC: корневая ФС у контейнера только на чтение.

### Зафиксированные зависимости

`requirements.in` содержит одну строку: ссылку на CPU-колесо vLLM.
`requirements.txt` получен из него через `uv pip compile` и содержит все
152 пакета (включая транзитивные) с точными версиями и sha256:

```bash
docker run --rm -v "$PWD/docker/vllm":/w -w /w ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim \
  uv pip compile requirements.in -o requirements.txt \
    --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match --generate-hashes --no-header
```

В Dockerfile установка идёт с `--require-hashes --no-deps`: если какого-то
пакета нет в списке или у файла другой хэш, сборка падает, а не тянет
что-то новое. После установки выполняется `pip check`.

### Логи без промптов (`privacy_logging`)

vLLM 0.32 пишет пользовательские данные в лог в трёх местах:

| Где | Что попадает в лог | Когда |
|---|---|---|
| `entrypoints/serve/utils/request_logger.py` | промпт, тело запроса (DEBUG), текст ответа, SamplingParams (INFO) | `--enable-log-requests`, `--enable-log-outputs` |
| `entrypoints/serve/middleware/log_response.py` | тело HTTP-ответа | `VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true` |
| `logging_utils/dump_input.py` | scheduler output с `prompt_token_ids` | при падении движка |

Решение: свой `logging_config.json` (vLLM подхватывает его из
`VLLM_LOGGING_CONFIG_PATH`) с фильтром `PromptRedactionFilter` на handler'е.
Фильтр стоит на handler, а не на логгере, поэтому через него проходят записи
всех процессов vLLM: и API-сервера, и EngineCore.

Для трёх перечисленных логгеров работает белый список:

* `Received request ...` переписывается в id запроса + безопасные параметры
  (`max_tokens`, `temperature`, `top_p`...). Stop-строки, json-схемы и
  logit_bias выкидываются, в них тоже бывает пользовательский текст.
* `Generated response ...` переписывается в id + длину ответа + finish_reason.
* Всё остальное от этих логгеров (DEBUG с промптом, тело ответа, дамп
  scheduler output) отбрасывается. Если в новой версии vLLM поменяется текст
  сообщения, оно пропадёт из лога, а не утечёт.

Чтобы факт обращения логировался, в ConfigMap включены
`enable-log-requests` и `enable-log-outputs`. Кроме того, uvicorn пишет
access-лог (метод, путь, статус). Пример того, что остаётся в логе:

```
(APIServer pid=1) INFO 10-27 15:14:44 [request_logger.py:65] Received request chatcmpl-b9e31c65ee9b1eeb: n=1, max_tokens=64, min_tokens=0, temperature=0.7, top_p=0.8, top_k=20, lora: False
(APIServer pid=1) INFO 10-27 15:14:44 [async_llm.py:578] Added request chatcmpl-b9e31c65ee9b1eeb-8b2826c8.
(APIServer pid=1) INFO 10-27 15:15:01 [request_logger.py:108] Generated response chatcmpl-b9e31c65ee9b1eeb: output_chars=140, finish_reason: length
(APIServer pid=1) INFO 10-27 15:15:01 [httptools_impl.py:482] 127.0.0.1:42018 - "POST /v1/chat/completions HTTP/1.1" 200
```

Проверка на живом поде: отправлял запросы (обычный и со `stream: true`) с
метками в промпте и в stop-строке, затем `kubectl logs --all-containers
... | grep -c` по меткам и по тексту ответа модели. Результат 0 в обоих
случаях, при этом строки `Received request` / `Generated response` с id
запроса на месте.

Тесты: `make test` (или `python -m pytest -q tests`), 14 штук: фильтр
логов и загрузчик модели (сеть не нужна, `snapshot_download` подменяется).
Те же тесты, hadolint, shellcheck и валидация манифестов через kubeconform
гоняются в GitHub Actions.

### Что всплыло при запуске

* **Модель и квантизация.** Проверял на ноутбучном Intel только с AVX2
  (без AVX512). Перебрал четыре варианта Qwen2.5-1.5B-Instruct, каждый
  переключается правкой `config/model.env` без пересборки образа:

  | Вариант | Что произошло | 64 токена ответа |
  |---|---|---|
  | bf16 (исходные веса) | oneDNN без AVX512-BF16 не умеет bf16 matmul и уходит в медленный запасной путь torch | 17–19 с |
  | fp32 | веса 8,7 ГБ, под KV-кэш остаётся 1 ГБ, нода minikube (12 ГБ) упирается в память | 53–125 с |
  | AWQ int4 (`Qwen/Qwen2.5-1.5B-Instruct-AWQ`) | падает: ядра `cpu_gemm_wna16` нет в AVX2-сборке vLLM (`_C_AVX2.abi3.so`), только в AVX512 | — |
  | **W8A8 int8** (`RedHatAI/...-quantized.w8a8`) | считается через `onednn_scaled_mm`, который есть в AVX2-сборке и использует AVX-VNNI | **9,3 с** |

  Остановился на W8A8. На сервере с AVX512-BF16 или AMX картина другая:
  там быстрее всего, скорее всего, будут bf16 или AWQ, и выбор стоит
  переделать.
* **Startup probe.** Первая компиляция на CPU идёт дольше 10 минут, и
  проба с бюджетом 10 минут убивала контейнер посреди компиляции. Бюджет
  поднят до 30 минут, а кэш компиляции вынесен на PVC.
* **xet.** Загрузка весов по протоколу xet на медленной сети minikube
  зависала без ошибки. В образе загрузчика стоит `HF_HUB_DISABLE_XET=1`,
  обычная HTTP-загрузка докачивается после обрыва.

### Модель на PVC и initContainer (бонус 2)

`docker/model-downloader` собирает образ на `python:3.12-alpine` (около 33 МБ
сжатым) с одной зависимостью `huggingface_hub`, тоже зафиксированной с хэшами.
Пользователь uid 10001.

`download_model.py` запускается initContainer'ом:

1. Если в `MODEL_DIR` есть маркер `.download-complete` с тем же `MODEL_ID` и
   `MODEL_REVISION`, скрипт сразу завершается, в сеть не ходит.
2. Иначе качает снапшот (`snapshot_download`, только `*.safetensors`,
   `*.json` и файлы токенизатора) и атомарно пишет маркер.
3. Если поменять ревизию в ConfigMap, докачаются только изменившиеся файлы.
4. После загрузки удаляет недокачанные `*.incomplete`, оставшиеся от
   оборванных попыток. С `PRUNE_OTHER_MODELS=true` удаляет с PVC прежние
   модели при смене модели (только каталоги со своим маркером, кэш
   компиляции не трогает).

Ревизия модели закреплена коммитом, а не `main`, чтобы веса не поменялись
сами при следующем скачивании.

Основной контейнер монтирует PVC только на чтение, запускается с
`HF_HUB_OFFLINE=1` и без маркера отказывается стартовать.

### Секреты

Один Secret `vllm-secrets` с двумя ключами:

* `HF_TOKEN` передаётся только в initContainer (`optional: true`, для
  открытых моделей не нужен);
* `VLLM_API_KEY` передаётся только в контейнер vLLM. vLLM сам читает эту
  переменную и требует `Authorization: Bearer ...` на всех `/v1/*`.
  `entrypoint.sh` не даёт стартовать с пустым токеном.

Secret собирается kustomize из `k8s/base/secrets.env`. Файл в `.gitignore`,
в репозитории лежит только `secrets.env.example`. В реальном кластере его
стоит заменить на External Secrets или Sealed Secrets.

### Конфигурация через ConfigMap

* `k8s/base/config/vllm-config.yaml` содержит аргументы `vllm serve`,
  монтируется файлом и передаётся через `--config`;
* `k8s/base/config/vllm.env` содержит переменные CPU-бэкенда
  (`VLLM_CPU_KVCACHE_SPACE`, `VLLM_CPU_OMP_THREADS_BIND`);
* `k8s/base/config/model.env` задаёт модель, ревизию и путь на PVC.

ConfigMap создаются `configMapGenerator`, к имени дописывается хэш
содержимого. Поменяли, например, `max-num-seqs` и сделали `kubectl apply -k
k8s/base`: Deployment ссылается на новое имя ConfigMap и перекатывается сам,
пересобирать образ не нужно.

### Безопасность пода

* `runAsNonRoot`, uid/gid 10001, `fsGroup` для записи на PVC;
* `readOnlyRootFilesystem`, запись только в emptyDir (`/tmp`, `/dev/shm`)
  и в каталог кэша компиляции на PVC (`~/.cache`);
* стратегия `Recreate`: PVC в режиме ReadWriteOnce, и на ноде нет памяти
  на две копии модели одновременно;
* `allowPrivilegeEscalation: false`, `capabilities: drop ALL`,
  `seccompProfile: RuntimeDefault`;
* namespace помечен Pod Security Standard `restricted`, с другими настройками
  под просто не создастся;
* `automountServiceAccountToken: false`, поду не нужен доступ к API кластера;
* NetworkPolicy (`k8s/base/networkpolicy.yaml`): в namespace всё запрещено
  по умолчанию. К vLLM на порт 8000 можно ходить только Job'у нагрузочного
  теста и подам из namespace'ов с меткой `llm-access=true`. Наружу под
  ходит только в DNS и на 443 (initContainer качает веса). Пробы kubelet и
  `kubectl port-forward` идут с ноды и политиками не блокируются.
  Политики работают только с CNI, который их поддерживает: в minikube это
  `minikube start --cni=calico`, со встроенным CNI они создаются, но не
  применяются.

## Нагрузочное тестирование (бонус 3)

```bash
make bench      # kubectl apply -k k8s/bench, ждёт Job и сохраняет лог
```

Job запускает `vllm bench serve` из того же образа внутри кластера
(без port-forward, который сам по себе добавляет задержку). Датасет `random`,
`--ignore-eos`, чтобы каждый ответ был ровно 128 токенов и прогоны были
сравнимы. Сценарии: короткий (128 токенов) и длинный (3072 токена) промпт,
1 и 8 параллельных клиентов. Запросов в каждом сценарии немного (5–16),
потому что на CPU один ответ занимает десятки секунд. Для медианы и
порядка величин этого достаточно, p95 на таких выборках ориентировочный.

Результаты для W8A8, 27.10.2026, minikube 12 CPU / 12 ГБ, Intel с AVX2
(полный лог: `bench/results/latest.log`):

| Сценарий | Вход/выход, ток. | Клиентов | Запросов | TTFT p50, мс | TTFT p95, мс | TPOT p50, мс | Latency p50, с | Latency p95, с | Throughput, req/s | Output, tok/s | Total, tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| short-c1 | 157/128 | 1 | 5 | 814 | 1097 | 152 | 20.2 | 20.4 | 0.05 | 6.3 | 14.1 |
| short-c8 | 157/128 | 8 | 16 | 2840 | 5369 | 192 | 28.3 | 29.4 | 0.28 | 36.2 | 80.6 |
| long-c1 | 3101/128 | 1 | 3 | 16315 | 16880 | 161 | 36.7 | 37.7 | 0.03 | 3.5 | 87.5 |
| long-c8 | 3101/128 | 8 | 8 | 33716 | 81298 | 710 | 123.9 | 124.3 | 0.06 | 8.2 | 207.7 |

Вход 157 токенов вместо 128, потому что `openai-chat` добавляет к промпту
шаблон чата Qwen. TTFT считается от отправки запроса до первого токена,
TPOT это время на каждый следующий токен, Latency это время всего ответа
целиком.

Что видно по цифрам:

* **Короткие промпты.** При 8 клиентах throughput вырос в 5,7 раза
  (6,3 → 36 ток/с), а TPOT ухудшился только с 152 до 192 мс. Декодирование
  упирается в чтение весов из памяти, и батч из 8 запросов читает их
  столько же раз, сколько один запрос. Это главный рычаг для сценария А.
* **Длинные промпты.** Prefill 3100 токенов занимает около 16 с
  (~190 ток/с). При 8 клиентах TTFT растёт до 34 с (p50) и 81 с (p95):
  промпты ждут друг друга в очереди на prefill, а TPOT вырастает до
  710 мс, потому что декодирование делит шаги с prefill чужих запросов
  (chunked prefill). Суммарно обработано 208 токенов в секунду, но
  отдельный пользователь это время ждёт.

Для сравнения тот же тест на исходной bf16-модели (до перехода на W8A8;
длинный сценарий с 8 клиентами не гонял, он шёл бы больше получаса):

| Сценарий | TTFT p50, мс | TPOT p50, мс | Latency p50, с | Output, tok/s |
|---|---|---|---|---|
| short-c1 | 5345 | 252 | 37.7 | 3.4 |
| short-c8 | 48966 | 442 | 125.7 | 8.1 |
| long-c1 | 139419 | 247 | 170.8 | 0.75 |

Лог: `bench/results/bf16/bench.log`. На этом процессоре переход на W8A8
ускорил prefill в 8,5 раза (TTFT длинного промпта 139 → 16 с), время на
токен в 1,7 раза (TPOT 252 → 152 мс), а выход при 8 клиентах в 4,5 раза.

## Как поднять Throughput

Рассуждение опирается на документацию vLLM (раздел Optimization and Tuning
и CPU backend) и на то, что видно в замерах выше. Параметры задаются в
`k8s/base/config/vllm-config.yaml` и `vllm.env`.

**А) Много запросов с небольшим числом токенов.** Узкое место здесь в
том, сколько последовательностей движок обрабатывает за один шаг
декодирования. На каждом шаге веса модели читаются из памяти один раз,
и чем больше запросов в батче, тем дешевле обходится каждый токен.

* `max-num-seqs`: поднять (16 → 64–128). Это максимальный размер батча,
  при коротких запросах он упирается именно в этот лимит, а не в память.
* `VLLM_CPU_KVCACHE_SPACE`: увеличить вместе с `max-num-seqs`, иначе
  запросы будут ждать свободных блоков KV-кэша или вытесняться.
* `max-num-batched-tokens` можно оставить умеренным (2048–4096): промпты
  короткие, и большой бюджет на prefill не нужен.
* prefix caching: в vLLM 0.32 он включён по умолчанию (в логе пода
  `enable_prefix_caching=True`), выключать не нужно. Если у запросов общий
  системный промпт, его KV-кэш переиспользуется, и prefill каждого запроса
  сокращается до уникальной части. В бенчмарке промпты случайные, поэтому
  там эффекта нет.
* `async-scheduling` на GPU как раз для этого случая, но CPU-платформа
  vLLM 0.32 принудительно его выключает (`platforms/cpu.py`), так что здесь
  не поможет.
* Горизонтально: на CPU несколько реплик по 4–8 ядер (с привязкой к ядрам
  одного NUMA-узла) обычно дают больше суммарной пропускной способности,
  чем одна реплика на все ядра: OpenMP-потоки одного процесса плохо
  масштабируются за пределы десятка ядер. Это стоит проверить замером на
  конкретном железе.

**Б) Мало запросов, но с огромным контекстом.** Здесь основное время
уходит на prefill (видно по TTFT длинного сценария), а память занимает
KV-кэш.

* `max-model-len`: выставить под реальный размер контекста. vLLM
  проверяет, что хотя бы один запрос максимальной длины помещается в
  KV-кэш, и лишний запас просто занимает память.
* `VLLM_CPU_KVCACHE_SPACE`: увеличить, чтобы одновременно поместилось
  несколько длинных запросов. Иначе они выполняются по очереди.
* `max-num-batched-tokens`: увеличить (8192–16384). Длинный промпт
  обрабатывается кусками этого размера (chunked prefill), и чем крупнее
  кусок, тем меньше шагов и эффективнее матричные операции. Цена:
  растёт задержка между токенами у тех, кто в это время декодирует.
* `long-prefill-token-threshold`: ограничивает, сколько токенов одного
  длинного промпта попадает в шаг, чтобы он не занимал весь бюджет и
  короткие запросы не стояли за ним в очереди.
* prefix caching (включён по умолчанию): при RAG или длинных документах,
  к которым задают несколько вопросов, общий префикс считается один раз.
  Чтобы это работало, документ должен стоять в начале промпта, а вопрос
  в конце.
* `kv-cache-dtype: fp8` (CPU-бэкенд его поддерживает): KV-кэш в два раза
  меньше, значит, помещается вдвое больший контекст или вдвое больше
  запросов. Немного страдает
  точность, нужно проверять на своих данных.
* `max-num-seqs` здесь можно уменьшить: большой батч всё равно не
  поместится в память, а лишние слоты только мешают планировщику.
