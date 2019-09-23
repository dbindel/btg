##############################################################
# btg - Bayesian Trans-Gaussian prediction
##############################################################

# Makefile

##############################################################

install: btgsrc
	cp src/btg .
	cp src/tkbtg.main .
	cp src/btghelp.txt .
	/bin/sh install.sh

install-sunperf:
	cp bin/btg-sunperf btg
	cp src/tkbtg.main .
	cp src/btghelp.txt .
	/bin/sh install.sh

install-alphaperf:
	cp bin/btg-alpha-dxml btg
	cp src/tkbtg.main .
	cp src/btghelp.txt .
	/bin/sh install.sh

btgsrc: btglapack
	cd src; make

btglapack:
	cd lapack; make
