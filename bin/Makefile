-include ../Makefile.local
-include ../Makefile.include

CFLAGS += -I../
CFLAGS += $(BORUVKA_CFLAGS)
CFLAGS += $(OPTS_CFLAGS)
CFLAGS += $(PROTOBUF_CFLAGS)
CFLAGS += $(NANOMSG_CFLAGS)
LDFLAGS += -L../ -lplan
LDFLAGS += $(BORUVKA_LDFLAGS)
LDFLAGS += $(OPTS_LDFLAGS)
LDFLAGS += $(PROTOBUF_LDFLAGS)
LDFLAGS += $(NANOMSG_LDFLAGS)
LDFLAGS += -pthread -lrt -lm
LDFLAGS += -lstdc++

ifeq '$(BIN_STATIC)' 'yes'
  LDFLAGS += -static
endif

OBJS  = search.o
OBJS += options.o

TARGETS = search

all: $(TARGETS)

search: ../libplan.a $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)
ma-search: ma-search.c ../libplan.a
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGETS)
	rm -f *.o

.PHONY: all
