include "globals.mzn";

int: n; 
array [1..n] of var 1..n: d;

constraint all_different(d);
constraint forall (i,j in 1..n where i!=j) (abs(d[i]-d[j]) != abs(i-j));

solve satisfy;

output ["d = \(d)\n"];
output	
  [if fix(d[i]) == j then "D " 
   else ". " 
   endif
   ++
   if j = n then "\n" 
   else "" endif	
   | i, j in 1..n];