VLLM_IMAGE       ?= vllm-cpu:0.32.0
DOWNLOADER_IMAGE ?= model-downloader:0.1.0
# Без плагина buildx docker build падает; Dockerfile совместимы с обоими сборщиками.
DOCKER_BUILD     := $(shell docker buildx version >/dev/null 2>&1 && echo docker build || echo DOCKER_BUILDKIT=0 docker build)

.PHONY: images load deploy test token logs bench clean

images:
	$(DOCKER_BUILD) -f docker/vllm/Dockerfile -t $(VLLM_IMAGE) .
	$(DOCKER_BUILD) -f docker/model-downloader/Dockerfile -t $(DOWNLOADER_IMAGE) .

load:
	minikube image load --overwrite $(VLLM_IMAGE)
	minikube image load --overwrite $(DOWNLOADER_IMAGE)

k8s/base/secrets.env:
	@printf 'HF_TOKEN=%s\nVLLM_API_KEY=%s\n' "$${HF_TOKEN}" "$$(openssl rand -hex 32)" > $@
	@echo "created $@"

deploy: k8s/base/secrets.env
	kubectl apply -k k8s/base
	kubectl -n llm rollout status deploy/vllm --timeout=20m

token:
	@kubectl -n llm get secret -l app.kubernetes.io/name=vllm -o jsonpath='{.items[0].data.VLLM_API_KEY}' | base64 -d; echo

logs:
	kubectl -n llm logs deploy/vllm -c vllm -f

test:
	python -m pytest -q tests

bench:
	kubectl -n llm delete job vllm-bench --ignore-not-found
	kubectl apply -k k8s/bench
	kubectl -n llm wait --for=condition=complete job/vllm-bench --timeout=60m
	kubectl -n llm logs job/vllm-bench > bench/results/latest.log

clean:
	kubectl delete -k k8s/base --ignore-not-found
