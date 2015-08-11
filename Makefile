###
# libplan
# --------
# Copyright (c)2014 Daniel Fiser <danfis@danfis.cz>
#
#  This file is part of boruvka.
#
#  Distributed under the OSI-approved BSD License (the "License");
#  see accompanying file BDS-LICENSE for details or see
#  <http://www.opensource.org/licenses/bsd-license.php>.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the License for more information.
##
-include Makefile.local
-include Makefile.include

CFLAGS += -I.
CFLAGS += $(BORUVKA_CFLAGS)
CFLAGS += $(OPTS_CFLAGS)

TARGETS  = libgnn.a

OBJS  = gsrm
OBJS += gng-t

OBJS := $(foreach obj,$(OBJS),.objs/$(obj).o)

all: $(TARGETS)

libgnn.a: $(OBJS)
	ar cr $@ $(OBJS)
	ranlib $@

.objs/%.o: src/%.c src/%.h
	$(CC) $(CFLAGS) -c -o $@ $<
.objs/%.o: src/%.c gnn/%.h
	$(CC) $(CFLAGS) -c -o $@ $<
.objs/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS)
	rm -f .objs/*.o
	rm -f $(TARGETS)
	if [ -d bin ]; then $(MAKE) -C bin clean; fi;
	if [ -d testsuites ]; then $(MAKE) -C testsuites clean; fi;
	if [ -d doc ]; then $(MAKE) -C doc clean; fi;
	
check:
	$(MAKE) -C testsuites check
check-valgrind:
	$(MAKE) -C testsuites check-valgrind
check-segfault:
	$(MAKE) -C testsuites check-segfault

doc:
	$(MAKE) -C doc

analyze: clean
	$(SCAN_BUILD) $(MAKE)

submodule:
	git submodule init
	git submodule update

third-party: submodule
	$(MAKE) -C third-party/boruvka
	$(MAKE) -C third-party/opts

help:
	@echo "Targets:"
	@echo "    all            - Build library"
	@echo "    doc            - Build documentation"
	@echo "    check          - Build & Run automated tests"
	@echo "    check-valgrind - Build & Run automated tests in valgrind(1)"
	@echo "    check-segfault - Build & Run automated tests in valgrind(1) set up to detect only segfaults"
	@echo "    clean          - Remove all generated files"
	@echo "    analyze        - Perform static analysis using Clang Static Analyzer"
	@echo "    submodule      - Fetch all submodules using git."
	@echo "    third-party    - Build all third-party projects."
	@echo "    third-party-clean - Clean all third-party projects."
	@echo ""
	@echo "Options:"
	@echo "    CC         - Path to C compiler          (=$(CC))"
	@echo "    M4         - Path to m4 macro processor  (=$(M4))"
	@echo "    SCAN_BUILD - Path to scan-build          (=$(SCAN_BUILD))"
	@echo "    PYTHON2    - Path to python v2 interpret (=$(PYTHON2))"
	@echo ""
	@echo "    DEBUG      'yes'/'no' - Turn on/off debugging   (=$(DEBUG))"
	@echo "    PROFIL     'yes'/'no' - Compiles profiling info (=$(PROFIL))"
	@echo ""
	@echo "Variables:"
	@echo "  Note that most of can be preset or changed by user"
	@echo "    SYSTEM            = $(SYSTEM)"
	@echo "    CFLAGS            = $(CFLAGS)"
	@echo "    LDFLAGS           = $(LDFLAGS)"
	@echo "    CONFIG_FLAGS      = $(CONFIG_FLAGS)"
	@echo "    USE_LOCAL_OPTS    = $(USE_LOCAL_OPTS)"
	@echo "    OPTS_CFLAGS       = $(OPTS_CFLAGS)"
	@echo "    OPTS_LDFLAGS      = $(OPTS_LDFLAGS)"
	@echo "    USE_LOCAL_BORUVKA = $(USE_LOCAL_BORUVKA)"
	@echo "    BORUVKA_CFLAGS    = $(BORUVKA_CFLAGS)"
	@echo "    BORUVKA_LDFLAGS   = $(BORUVKA_LDFLAGS)"

.PHONY: all clean check check-valgrind help doc install analyze examples submodule third-party
