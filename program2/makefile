all: disposable persistent

disposable: patel_shiny_disposable.o def.o
	gcc -pthread -o disposable patel_shiny_disposable.o def.o -O3 -lrt
	
disposable.o: patel_shiny_disposable.c def.h
	gcc -c -pthread patel_shiny_disposable.c -O3 -lrt 

persistent: patel_shiny_persistent.o def.o
	gcc -pthread -o persistent patel_shiny_persistent.o def.o -O3 -lrt
	
persistent.o: patel_shiny_persistent.c def.h
	gcc -c -pthread patel_shiny_persistent.c -O3 -lrt 

def.o: def.c def.h
	gcc -c def.c -O3 -lrt 