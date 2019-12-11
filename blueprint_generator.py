import copy
import random
import numpy as np

#call this with generate_layout(layout,0,height-1,0,width-1, 0)
MAX_DEPTH = 5
BUILD_CHANCE = 0.5

def generate_layout(layout, row_start, row_end, col_start, col_end, signal):
    #base case 1: if rect size too small, do not recur
    if (row_end-row_start)<2 or (col_end-col_start)<2:
        return
    #base case 2: if last signal is (-) and we do not construct again, do not recur
    #based on depth signal create rectangle
    build_val = random.random()
    if build_val >= BUILD_CHANCE:
        construct(layout, row_start, row_end, col_start, col_end)
    #if this call doesn't create a rectangle, negate the depth signal-> base case
    #randomize split point
    options = ['n']
    if (row_end-row_start)>=4:
        options.append('h')
    if (col_end-col_start)>=4:
        options.append('v')
    split_type = random.choice(options)
    #recur on two halves based on random values
    if split_type == 'h':
        split_point = random.randint(row_start+2, row_end-2)
        generate_layout(layout, row_start, split_point, col_start, col_end, signal)
        generate_layout(layout, split_point, row_end, col_start, col_end, signal)
    if split_type == 'v':
        split_point = random.randint(col_start+2, col_end-2)
        generate_layout(layout, row_start, row_end, split_point, col_end, signal)
        generate_layout(layout, row_start, row_end, col_start, split_point, signal)

def construct(layout, row_start, row_end, col_start, col_end):
    for i in range(row_start,row_end+1):
        layout[i][col_start],layout[i][col_end] = 1,1
    for i in range(col_start,col_end+1):
        layout[row_start][i],layout[row_end][i] = 1,1


def expand_layout(layout, height):
    if height<3:
        return None     
    blueprint = []
    for h in range(height):
        blueprint.append(layout)
    blueprint[0] = copy.deepcopy(blueprint[0])
    blueprint[-1] = blueprint[0]
    fill(blueprint[0])
    return blueprint

def fill(layout):
    #np.array( (X.cumsum(1) > 0) & (np.flip(np.flip(X, 1).cumsum(1), 1) > 0), dtype='int' )
    for r,row in enumerate(layout):
        for c,tile in enumerate(row):
            if tile == 0 and getTile(layout,r-1,c)==1 and getTile(layout,r,c-1)==1:
                fill_subgrid(layout,r,c)

def fill_subgrid(layout,row,col):
    row_found, col_found = False,False
    row_sz,col_sz = 1,1
    while not row_found:
        cur = getTile(layout,row+row_sz,col+1)
        if cur is None:
            row_found = True
            row_sz = -1
        elif cur == 1:
            row_found = True
        else:
            row_sz += 1
    while not col_found:
        cur = getTile(layout,row+1,col+col_sz)
        if cur is None:
            col_found = True
            col_sz = -1
        elif cur == 1:
            col_found = True
        else:
            col_sz += 1
    if row_sz!=-1 and col_sz!=-1:
        for r in range(0,row_sz):
            for c in range(0,col_sz):
                layout[row+r][col+c] = 1


def getTile(layout,row,col):
    if 0<=row<len(layout) and 0<=col<len(layout[0]):
        return layout[row][col]
    return None

def generate_blueprint(length,width,height):
    base = [[0 for i in range(width)] for i in range(length)]
    generate_layout(base, 0, length-1, 0, width-1, 0)
    return expand_layout(base,height)

def generate_1d_blueprint(length,width,height):
    base = [[0 for i in range(width)] for i in range(length)]
    empty = [[0 for i in range(width)] for i in range(length)]
    generate_layout(base, 0, length-1, 0, width-1, 0)
    bp = [base]
    for h in range(height-1):
        bp.append(empty)
    return bp
#   blueprint = generate_1d_blueprint(10,16,4)


