-include ../Makefile.local
-include ../Makefile.include

CFLAGS += -I../
CFLAGS += $(BORUVKA_CFLAGS)
CFLAGS += $(OPTS_CFLAGS)
LDFLAGS += -L../ -lgnn
LDFLAGS += $(BORUVKA_LDFLAGS)
LDFLAGS += $(OPTS_LDFLAGS)
LDFLAGS += -pthread -lrt -lm

ifeq '$(BIN_STATIC)' 'yes'
  LDFLAGS += -static
endif

OBJS =

TARGETS = gnn-gsrm

all: $(TARGETS)

gnn-%: %.c ../libgnn.a 
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGETS)
	rm -f *.o

.PHONY: all
