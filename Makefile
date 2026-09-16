# ================================================================================
# GRU DDOS DETECTION CICDDOS2019 MAKEFILE
# ================================================================================
# Author      : Breno Farias da Silva
# Created     : 2026-09-14
# Description :
#     Automates virtual-environment creation, dependency installation, platform
#     validation, and experiment execution for macOS Apple Silicon and Linux GPU
#     servers. Runtime output is mirrored by main.py to logs/main.log.
#
#     Key features include:
#         - Separate run-mac and run-linux experiment targets.
#         - Overridable sampling, batch-size, dataset, and output variables.
#         - Optional detached SSH execution through DETACH=1.
#
# Usage:
#     1. Run make run-mac on Apple Silicon or make run-linux on Linux.
#     2. Override variables inline, for example: make run-linux BATCH_SIZE=512.
#     3. Use make tail-log to follow logs/main.log.
#
# Outputs:
#     - Creates .venv and installs requirements.txt dependencies.
#     - Writes runtime output through main.py to logs/main.log.
#     - Writes experiment artifacts inside the configured project output directory.
#
# TODOs:
#     - None identified.
#
# Dependencies:
#     - GNU Make or a compatible make implementation.
#     - Python 3.11 or 3.12.
#     - Platform dependencies declared in requirements.txt.
#
# Assumptions & Notes:
#     - Linux GPU execution requires an NVIDIA driver visible to TensorFlow.
#     - macOS playback uses afplay; Linux playback optionally uses aplay.
# ================================================================================

SHELL := /bin/bash
PROJECT_DIR := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_DIR)/.venv
PYTHON := $(VENV)/bin/python
PYTHON_CMD ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null)
LOG_DIR := $(PROJECT_DIR)/logs
LOG_FILE := $(LOG_DIR)/main.log
PID_FILE := $(LOG_DIR)/main.pid
UNAME_S := $(shell uname -s 2>/dev/null)

MAC_DATA_DIR ?= /Users/brenofarias/Downloads/RAW Datasets/CICDDoS2019
LINUX_DATA_DIR ?= $(HOME)/RAW Datasets/CICDDoS2019
MAC_OUTPUT_DIR ?= $(PROJECT_DIR)/GRU-DDoS-Detection-100k
LINUX_OUTPUT_DIR ?= $(PROJECT_DIR)/GRU-DDoS-Detection-Full

MAC_SAMPLING_PROFILE ?= cap-per-class
MAC_PER_CLASS_CAP ?= 100000
LINUX_SAMPLING_PROFILE ?= all
LINUX_PER_CLASS_CAP ?= 100000
MAC_BATCH_SIZE ?= 256
LINUX_BATCH_SIZE ?= 1000
CHUNKSIZE ?= 50000
SPLIT_SEED ?= 42
MODEL_SEED ?= 1337
GRU_UNITS ?= 8
DENSE1 ?= 16
DENSE2 ?= 8
DROPOUT ?= 0.10
LEARNING_RATE ?= 0.001
EPOCHS ?= 100
EARLY_MIN_DELTA ?= 0.001
EARLY_PATIENCE ?= 5
RUNS ?= 1

COMMON_ARGS = --source-day "01-12" --chunksize "$(CHUNKSIZE)" --split-seed "$(SPLIT_SEED)" --model-seed "$(MODEL_SEED)" --no-stratify --scaling-mode "separate" --validation-mode "test" --gru-units "$(GRU_UNITS)" --dense1 "$(DENSE1)" --dense2 "$(DENSE2)" --dropout "$(DROPOUT)" --learning-rate "$(LEARNING_RATE)" --epochs "$(EPOCHS)" --early-min-delta "$(EARLY_MIN_DELTA)" --early-patience "$(EARLY_PATIENCE)" --runs "$(RUNS)"
MAC_ARGS = --data-dir "$(MAC_DATA_DIR)" --output-dir "$(MAC_OUTPUT_DIR)" --sampling-profile "$(MAC_SAMPLING_PROFILE)" --per-class-cap "$(MAC_PER_CLASS_CAP)" --batch-size "$(MAC_BATCH_SIZE)" $(COMMON_ARGS)
LINUX_ARGS = --data-dir "$(LINUX_DATA_DIR)" --output-dir "$(LINUX_OUTPUT_DIR)" --sampling-profile "$(LINUX_SAMPLING_PROFILE)" --per-class-cap "$(LINUX_PER_CLASS_CAP)" --batch-size "$(LINUX_BATCH_SIZE)" $(COMMON_ARGS)

ifeq ($(UNAME_S),Darwin)
run: run-mac
else ifeq ($(UNAME_S),Linux)
run: run-linux
else
run:
	@echo "Unsupported operating system: $(UNAME_S). Use macOS or Linux." >&2
	@exit 1
endif

all: run
mac: run-mac
linux: run-linux

run-mac: dependencies verify-mac prepare-runtime
	$(call RUN_EXPERIMENT,$(MAC_ARGS))

run-linux: dependencies verify-linux prepare-runtime
	$(call RUN_EXPERIMENT,$(LINUX_ARGS))

prepare-runtime:
	@mkdir -p "$(LOG_DIR)"
	@rm -f "$(PID_FILE)"

$(PYTHON):
	@if [ -z "$(PYTHON_CMD)" ]; then echo "Python 3.11 or 3.12 is required." >&2; exit 1; fi
	@echo "Creating virtual environment with $(PYTHON_CMD)..."
	@"$(PYTHON_CMD)" -m venv "$(VENV)"
	@"$(PYTHON)" -m pip install --upgrade pip setuptools wheel

dependencies: $(PYTHON)
	@echo "Installing/updating platform-aware Python dependencies..."
	@"$(PYTHON)" -m pip install -r "$(PROJECT_DIR)/requirements.txt"

verify-mac: dependencies
	@"$(PYTHON)" -c "import platform, tensorflow as tf; assert platform.system() == 'Darwin', 'run-mac requires macOS'; devices=tf.config.list_physical_devices('GPU'); print('TensorFlow:', tf.__version__); print('GPUs:', devices); assert devices, 'No TensorFlow Metal GPU detected.'"

verify-linux: dependencies
	@"$(PYTHON)" -c "import platform, tensorflow as tf; assert platform.system() == 'Linux', 'run-linux requires Linux'; devices=tf.config.list_physical_devices('GPU'); print('TensorFlow:', tf.__version__); print('GPUs:', devices); assert devices, 'No TensorFlow GPU detected on the Linux server.'"

define RUN_EXPERIMENT
	@if [ -z "$(DETACH)" ]; then \
		cd "$(PROJECT_DIR)" && "$(PYTHON)" main.py $(1); \
	else \
		cd "$(PROJECT_DIR)"; \
		nohup "$(PYTHON)" main.py $(1) >/dev/null 2>&1 & \
		pid=$$!; echo $$pid > "$(PID_FILE)"; \
		echo "Started detached process PID $$pid"; \
		echo "Log file: $(LOG_FILE)"; \
	fi
endef

tail-log:
	@mkdir -p "$(LOG_DIR)"
	@touch "$(LOG_FILE)"
	@tail -f "$(LOG_FILE)"

stop:
	@if [ ! -f "$(PID_FILE)" ]; then echo "No detached PID file found at $(PID_FILE)."; exit 1; fi
	@pid=$$(cat "$(PID_FILE)"); kill $$pid; rm -f "$(PID_FILE)"; echo "Stopped PID $$pid."

show-config:
	@echo "PROJECT_DIR=$(PROJECT_DIR)"
	@echo "MAC_DATA_DIR=$(MAC_DATA_DIR)"
	@echo "LINUX_DATA_DIR=$(LINUX_DATA_DIR)"
	@echo "MAC_OUTPUT_DIR=$(MAC_OUTPUT_DIR)"
	@echo "LINUX_OUTPUT_DIR=$(LINUX_OUTPUT_DIR)"
	@echo "MAC_SAMPLING_PROFILE=$(MAC_SAMPLING_PROFILE)"
	@echo "LINUX_SAMPLING_PROFILE=$(LINUX_SAMPLING_PROFILE)"
	@echo "MAC_BATCH_SIZE=$(MAC_BATCH_SIZE)"
	@echo "LINUX_BATCH_SIZE=$(LINUX_BATCH_SIZE)"
	@echo "CHUNKSIZE=$(CHUNKSIZE)"
	@echo "EPOCHS=$(EPOCHS)"
	@echo "RUNS=$(RUNS)"

clean-logs:
	@rm -f "$(LOG_DIR)"/*.log "$(PID_FILE)"

clean:
	@rm -rf "$(VENV)"
	@find "$(PROJECT_DIR)" -type f -name '*.pyc' -delete
	@find "$(PROJECT_DIR)" -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -f "$(PID_FILE)"

help:
	@echo "make run             Auto-select run-mac or run-linux"
	@echo "make run-mac         Bounded Apple-Silicon execution"
	@echo "make run-linux       Full-source Linux GPU execution"
	@echo "make run-linux DETACH=1   Detached SSH execution"
	@echo "make tail-log        Follow logs/main.log"
	@echo "make stop            Stop the PID started with DETACH=1"
	@echo "make show-config     Print effective configurable values"
	@echo "make clean           Remove .venv and Python caches"

.PHONY: all run mac linux run-mac run-linux prepare-runtime dependencies verify-mac verify-linux tail-log stop show-config clean-logs clean help
