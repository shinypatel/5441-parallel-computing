GCC = gcc
FLAGS = -O3
LIBS = -lrt -fopenmp

all: disposable persistent

disposable:
	$(GCC) $(FLAGS) $(LIBS) -o disposable patel_shiny_disposable.c def.c

persistent:
	$(GCC) $(FLAGS) $(LIBS) -o persistent patel_shiny_persistent.c def.c