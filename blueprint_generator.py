import copy

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
    for r,row in enumerate(layout):
        for c,tile in enumerate(row):
            if tile == 0 and getTile(layout,r-1,c)==1 and getTile(layout,r,c-1)==1:
                fill_subgrid(layout,r,c)

def fill_subgrid(layout,row,col):
    row_found, col_found = False,False
    row_sz,col_sz = 1,1
    while not row_found:
        cur = getTile(layout,row+col_sz,col)
        if cur is None:
            row_found = True
            col_sz = -1
        elif cur == 1:
            row_found = True
        else:
            col_sz += 1
    while not col_found:
        cur = getTile(layout,row,col+row_sz)
        if cur is None:
            col_found = True
            row_sz = -1
        elif cur == 1:
            col_found = True
        else:
            row_sz += 1
    if row_sz!=-1 and col_sz!=-1:
        for r in range(1,row_sz):
            for c in range(1,col_sz):
                layout_

'''
def dfs_fill(layout,row,col):
    if layout[row][col] == 0:
        layout[row][col] = 1
        adj = get_adjacent(layout,row,col)
        for a in adj:
            dfs_fill(layout,a[0],a[1])

def get_adjacent(layout,row,col):
    r_vals = [row,row,row+1,row-1]
    c_vals = [col+1,col-1,col,col]
    adj = []
    for i in range(4):
        if getTile(layout,r_vals[i],c_vals[i]):
            adj.append((r_vals[i],c_vals[i]))
    return adj
'''
def getTile(layout,row,col):
    if 0<=row<len(layout) and 0<=col<len(layout[0]):
        return layout[row][col]
    return None

layout = [[0,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,1,1,1,1,1,1,0,0,0,0],
          [0,1,0,0,0,0,0,1,0,0,0,0],
          [0,1,0,0,0,0,0,1,1,1,1,0],
          [0,1,0,0,0,0,0,1,0,0,1,0],
          [0,1,0,0,0,0,0,1,0,0,1,0],
          [0,1,0,0,0,0,0,1,1,1,0,0],
          [0,1,0,0,0,0,0,1,0,1,0,0],
          [0,1,1,1,1,1,1,1,1,1,0,0]]

blueprint = expand_layout(layout,4)
for i in blueprint[0]:
    print(i)