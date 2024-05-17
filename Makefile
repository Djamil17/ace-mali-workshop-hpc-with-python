.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.vhpc:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .vhpc
	. .vhpc/bin/activate; .vhpc/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .vhpc/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

vhpc: .venv

# setup
test: .vhpc
	. .vhpc/bin/activate; python3 -m

clean: .vhpc
	rm -rf .vhpc
