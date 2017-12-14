GCC = nvcc
FLAGS = -O
LIBS = -lm

all: lab4p1 lab4p2

lab4p2:
	$(GCC) $(FLAGS) $(LIBS) -o lab4p2 nvcc_bmpReader.o patel_shiny_lab4p2.cu 
	
lab4p1:
	$(GCC) $(FLAGS) $(LIBS) -o lab4p1 patel_shiny_lab4p1.cu 
	
clean:
	rm lab4p1 lab4p2