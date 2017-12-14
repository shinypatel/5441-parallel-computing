a.out: program1.o def.o
	gcc program1.o def.o -O3 -lrt
	
program1.o: program1.c def.h
	gcc -c program1.c -O3 -lrt 

def.o: def.c def.h
	gcc -c def.c -O3 -lrt 