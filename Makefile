.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION := $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.vhpc:
	python3 -m venv .vhpc
	. .vhpc/bin/activate; .vhpc/bin/pip3 install --upgrade pip3 ; .vhpc/bin/pip3 install -e .[dev,test]

vhpc: .venv

# setup
test: .vhpc
	. .vhpc/bin/activate; python3 -m

clean: .vhpc
	rm -rf .vhpc
