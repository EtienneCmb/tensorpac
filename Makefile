
# clean dist
clean_dist:
	@rm -rf build/
	@rm -rf build/
	@rm -rf tensorpac.egg-info/
	@rm -rf dist/
	@echo "Dist cleaned"

# build dist
build_dist: clean_dist
	python setup.py sdist
	python setup.py bdist_wheel
	@echo "Dist built"

# check distribution
check_dist:
	twine check dist/*

# upload distribution
upload_dist:
	twine upload --verbose dist/*
