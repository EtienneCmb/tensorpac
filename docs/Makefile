# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Tensorpac
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

install:
	@echo "Deploying the doc of tensorpac"
	rm -rf _build/doctrees _build/tensorpac.github.io
	# Credit to Mainak Jas for this part of code
	# first clone the EtienneCmb/tensorpac.github.io repo because it may ask
	# for password and we don't want to delay this long build in
	# the middle of it
	# --no-checkout just fetches the root folder without content
	# --depth 1 is a speed optimization since we don't need the
	# history prior to the last commit
	# -b gh-pages fetches only the branch for the gh-pages
	git clone -b gh-pages --single-branch --no-checkout --depth 1 https://github.com/EtienneCmb/tensorpac _build/tensorpac.github.io
	touch _build/tensorpac.github.io/.nojekyll
	make html
	cd _build/ && \
	cp -r html/* tensorpac.github.io && \
	cd tensorpac.github.io && \
	git add * && \
	git add .nojekyll && \
	git add CNAME && \
	git commit -a -m 'Make install' && \
	git push