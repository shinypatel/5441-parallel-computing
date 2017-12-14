// if the macro INCLUDE is not defined then include the code between #ifndef and #endif
#ifndef INCLUDE
#define INCLUDE
   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <time.h>
   #define NS_PER_US 1000
   
   struct box {
      /* *t_id: top neighbor ids
         *b_id: bottom neighbor ids
         *l_id: left neighbor ids
         *r_id: right neighbor ids */
      int info[6], top, bottom, left, right, *t_id, *b_id, *l_id, *r_id; 
      /* info contains upper-left x, upper-left y, height, width, upper-right x and
         upper-right y respectively where x and y are in matrix representation
         i.e. x represents row and y column */
      double temp;
   };

   struct timespec start, end;

   // convert a line into an int array
   int* split(char line[]);

   // splice an array and return it
   int* splice(int size, int *arr);

   // find maximum temp 
   double maximum(int i, double max, double temp);

   // find minimum temp
   double minimum(int i, double min, double temp);

   /* finds contact distance between a box and its neighbor
      x: width or height
      a: upper-left (x or y) of neighbor
      b: upper-right (x or y) of neighbor
      c: upper-left (x or y) of current box
      d: upper-right (x or y) of current box */
   int contactDistance(int x, int a, int b, int c, int d);
#endif