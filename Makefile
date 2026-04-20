# colors
ESC := \033
RED := $(ESC)[31m
GREEN := $(ESC)[32m
BOLD := $(ESC)[1m
RESET := $(ESC)[0m

# helpers
FLAKE8_SUCCESS := printf '%b\n' "$(BOLD)$(GREEN)Success: flake8$(RESET)"

# structure
DIRS := . src
ARGS ?=

PYCACHES := $(addsuffix /__pycache__,$(DIRS))
MYPYCACHES := $(addsuffix /.mypy_cache,$(DIRS))

# tools
UV := uv
FLAKE8 := $(UV) run flake8 --exclude=.venv,vllm-0.10.1,__pycache__,.mypy_cache
MYPY := $(UV) run mypy --exclude '(.*cache.*)'

# flags
MYPY_FLAGS := \
		--check-untyped-defs \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--warn-return-any \
		--disallow-untyped-defs

# user rules
install: uv
	@$(UV) sync

run: install
	@$(UV) run python -m src $(ARGS)

clean:
	@rm -rf $(PYCACHES) $(MYPYCACHES) .venv data/output

debug: install
	@$(UV) run python -m pdb -m src $(ARGS)

lint: install
	@$(FLAKE8) src && $(FLAKE8_SUCCESS)
	@$(MYPY) src $(MYPY_FLAGS)

# internal rules
uv:
	@command -v $(UV) >/dev/null 2>&1 \
	|| (printf "$(RED)Error: cannot use or find uv$(RESET)\n" \
	&& exit 1)

# miscellaneous
.PHONY: install run clean debug lint uv
