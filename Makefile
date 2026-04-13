SHELL = /bin/bash
.DEFAULT_GOAL = help

# Windows has different commands in shell
# - `RM` command: use for removing files
ifeq ($(OS), Windows_NT) 
    RM = del /Q /F
    PATH_SEP = \\
else
    RM = rm -rf
    PATH_SEP = /
endif

# Includes `docs/build`
DOCS_BUILD_FILES = docs$(PATH_SEP)build
# Includes `_output`, `test/_output`
TEST_OUTPUT_FILES = _output test$(PATH_SEP)_output
# Includes all the above
ALL_TMP_FILES = $(DOCS_BUILD_FILES) $(TEST_OUTPUT_FILES)

.PHONY: docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.precompile();'

docs: doc_init ## Generate documentation
	julia --startup-file=no --project=docs/ docs/make.jl

docs-serve: doc_init ## Serve documentation locally for preview in browser
	julia --project=docs/ -e 'ENV["DOCS_DRAFT"]="true"; using LiveServer; LiveServer.servedocs(launch_browser=true, port=5678)'

.PHONY: test

test: ## Run tests
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test()'

clean: ## Clean documentation build and test output
	$(foreach file, $(ALL_TMP_FILES), $(RM) $(file))

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
