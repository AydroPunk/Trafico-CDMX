# Atajos. Equivalen a escribir los comandos de uv completos.
#   make test              corre las pruebas
#   make run               corre todo el proyecto
#   make run PASO=exp2     corre un solo paso (silver, gold, exp1, exp2, exp3, figuras, manifiesto)

PASO ?= todo

.PHONY: test run

test:
	uv run pytest

run:
	uv run python -m trafico $(PASO)