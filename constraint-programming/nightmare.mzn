
include "globals.mzn"; 
   
% Parameters of the problems
int: n = 5;
array[1..n] of int: left_vision = [2,1,3,2,3];
array[1..n] of int: right_vision = [3,4,1,3,2];
array[1..n] of int: up_vision = [2,1,3,2,2];
array[1..n] of int: down_vision = [3,4,2,1,2];

% Variable: value for each cell
array[1..n,1..n] of var 1..n: city;

% 1: different size for all towers on the same line/column
constraint forall(i in 1..n, j in 1..n) (all_different(city[i,..]) /\ all_different(city[..,j]));
%constraint forall(i in 1..n) (all_different(city[i,..]));
%constraint forall(j in 1..n) (all_different(city[..,j]));

% left vision
constraint forall(i in 1..n, j in 1..1) (count(x in city[i,..])(x >= city[i,1]) = left_vision[i]);

% right vision
constraint forall(i in 1..n, j in n..n) (count(x in city[i,..])(x >= city[i,n]) = right_vision[i]);

% up vision
constraint forall(i in 1..1, j in 1..n) (count(x in city[..,j])(x >= city[1,j]) = up_vision[j]);

% down vision
constraint forall(i in n..n, j in 1..n) (count(x in city[..,j])(x >= city[n,j]) = down_vision[j]);

solve satisfy;

output
["city:"]++
[
  if j = 1 then "\n" else " " endif ++
    show(city[i,j])
  | i in 1..n, j in 1..n
];
