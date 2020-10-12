include "globals.mzn"; 

int: n;
array[1..n, 1..n] of var 0..1: grid;
array[1..n, 1..n] of var 0..1: x;
constraint forall(i in 1..n, j in 1..n)(x[i,j] = grid[i,j]);

% 1
constraint forall(i in 1..n, j in 1..n)(sum(x[i,..]) = n/2 /\ sum(x[..,j]) = n/2);

% 2, 3
constraint forall(i in 1..n, j in 1..n where i+2 <= n) (sum(x[i..i+2,j]) < 3 /\ sum(x[i..i+2,j]) > 0);
constraint forall(i in 1..n, j in 1..n where j+2 <= n) (sum(x[i,j..j+2]) < 3 /\ sum(x[i,j..j+2]) > 0);

% 4, 5
constraint forall(i in 1..n, j in 1..n where i+j <= n) (x[i,..] != x[i+j,..]);
constraint forall(i in 1..n, j in 1..n where j+i <= n) (x[..,j] != x[..,j+i]);

output [
   if j = 1 then "\n" else " " endif ++
     show(x[i,j])
   | i,j in 1..n
];