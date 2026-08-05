.PHONY: install test eval improve demo api web-cache clean

VENV ?= ./.venv/bin

install:            ## editable install with all extras (demo + test + eval + api)
	$(VENV)/pip install -e ".[demo,test,eval,api]"

test:               ## run the full test suite
	$(VENV)/pytest -q

eval:               ## labeled + held-out scoreboards and the LLM-as-judge
	$(VENV)/governor evaluate

improve:            ## the eval-gated self-improvement loop (deterministic proposer)
	$(VENV)/governor improve

demo:               ## the Streamlit dashboard
	$(VENV)/streamlit run app.py

api:                ## run the Governor as an HTTP service (needs the api extra)
	$(VENV)/uvicorn governor.api:app --reload

web-cache:          ## regenerate the offline web-demo cache
	$(VENV)/python scripts/gen_web_cache.py

clean:              ## remove build/test caches
	rm -rf .pytest_cache **/__pycache__ *.egg-info src/*.egg-info
