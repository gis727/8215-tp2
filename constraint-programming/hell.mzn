
% SOLVER: Chuffed

include "globals.mzn";

int: n_object;
array[1..n_object,1..2] of int: shape; % (x,y) shape of each object
int: max_length;
int: max_width;

int: max_x = sum(i in 1..n_object) (shape[i,1]) - min(i in 1..n_object) (shape[i,1]);
int: max_y = sum(i in 1..n_object) (shape[i,2]) - min(i in 1..n_object) (shape[i,2]);

array[1..n_object,1..2] of var 0..max(max_x, max_y) div 4: positions; % position of each rectangle bottom-left point. [rect1_x,rect1_y, rect2_x,rect2_y, ...]

% objects stay in the area
constraint forall(i in 1..n_object) (
positions[i,1] + shape[i,1] <= max_width /\
positions[i,2] + shape[i,2] <= max_length
);

% all objects do not overlap
constraint forall(i in 1..n_object) ( forall(j in 1..n_object where i != j) (
positions[j,1] + shape[j,1] <= positions[i,1] \/
positions[i,2] + shape[i,2] <= positions[j,2] \/
positions[i,1] + shape[i,1] <= positions[j,1] \/
positions[j,2] + shape[j,2] <= positions[i,2]
));

% define red rectangle dimensions
var int: rect_w = max(i in 1..n_object) (positions[i,1] + shape[i,1]) - min(i in 1..n_object) (positions[i,1]);
var int: rect_l = max(i in 1..n_object) (positions[i,2] + shape[i,2]) - min(i in 1..n_object) (positions[i,2]);
var int: rect_area = rect_w * rect_l;

% minimize red rectangle
solve minimize rect_area;