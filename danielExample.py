import numpy as np
import matplotlib.pyplot as plt
import math
# This script generates a very simple parachute geometry for AEROS. It was originally written by Daniel
# Huang and modified by me (Taggart Durrant) to generalize the number of nodes in the z direction, turning
# a quasi-2D parachute into a 3D one. Some of the comments and functionalities are obsolete now.

# User-defined parameters
nz = 81 # Number of nodes in the z direction
cl_z = 0.02 # Characteristic length of the mesh in the z direction
cl_x = 0.005 # Characteristic length of the mesh in the x direction
num_flip_diag = 10  # Number of quads before flipping diagonal
canopy_topology = 129 # Topology ID for the canopy elements. See AEROS documentation for details.
mPoints = 47  # number of cable segments

# NOTE: To change from straight line to a different shape, scroll down past the shape functions and change
# the function call to the desired shape function. 


# ============= Shape Functions ======================================================================

# This is a sin curve
# int  nPoints, number of nodes
# int* xArray, x coordinates of these nodes
# int* yArray, y coordinates of these nodes
def sinCurve(plotOrNot, xscale = 0.5, yscale = 0.5):
    nPoints = 800

    A = 0.2
    k = 4
    xArray = np.linspace(-0.5,0.5,num=nPoints)
    yArray = A*np.sin((2*k)*np.pi*(xArray + 0.5)-0.5*np.pi)
    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray*xscale, yArray*yscale

#This is a straight line
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def straightLine():
    nPoints = 21
    xArray = np.linspace(-0.8,0.8,num=nPoints)
    yArray = 0*xArray
    return nPoints, xArray, yArray



#This is a hat shape function
#int  nPoints, number of nodes
#int* xArray, x coordinates of these nodes
#int* yArray, y coordinates of these nodes
def hatShape():
    k = 25
    nPoints = 301
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    xArray[0:k+1] = np.linspace(0,0.25,num=k+1)
    yArray[0:k+1] = 0

    xArray[k:2*k+1] = 0.25
    yArray[k:2*k+1] = np.linspace(0,0.25,num=k+1)

    xArray[2*k:3*k+1] = np.linspace(0.25,0,num=k+1)
    yArray[2*k:3*k+1] = 0.25

    xArray[3*k:4*k+1] = 0
    yArray[3*k:4*k+1] = np.linspace(0.25,0.5,num=k+1)

    xArray[4*k:8*k+1] = np.linspace(0,1,num=4*k+1)
    yArray[4*k:8*k+1] = 0.5

    xArray[8*k:9*k+1] = 1
    yArray[8*k:9*k+1] = np.linspace(0.5,0.25,num=k+1)

    xArray[9*k:10*k+1] = np.linspace(1,0.75,num=k+1)
    yArray[9*k:10*k+1] = 0.25

    xArray[10*k:11*k+1] = 0.75
    yArray[10*k:11*k+1] = np.linspace(0.25,0,num=k+1)

    xArray[11*k:12*k+1] = np.linspace(0.75,1,num=k+1)
    yArray[11*k:12*k+1] = 0

    return nPoints, xArray, yArray

# This is a candle shape function
def candle(xscale, yscale):
    xx = np.array([-0.5, -0.07, -0.05, -0.45, -0.5, -0.05, -0.04, -0.5, -0.48, -0.38, -0.375, -0.25, -0.245, -0.13, -0.125, -0.05, -0.045, 0.045, 0.05, 0.125, 0.13, 0.245, 0.25,
          0.375, 0.38, 0.48, 0.5, 0.04, 0.05, 0.5, 0.45, 0.05, 0.07, 0.5]);
    yy = np.array([0, 0, 0.125, 0.125, 0.25, 0.25, 0.375, 0.375, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0, 0]);

    num = len(xx)
    return num, xx*xscale, yy*yscale


# This is a Hilbert curve generator
def hilbertCurveRecursive(n):
    ''' Generate Hilbert curve . 'n' must be a power of two. '''
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbertCurveRecursive(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))
def hilbertCurve(order,scaleX, scaleY):

    n = 2**order;
    idx = hilbertCurveRecursive(n)
    idx = np.rot90(idx,-1)
    idx = np.fliplr(idx)
    y, x = np.indices(idx.shape).reshape(2, -1) /(n-1.0)
    x = x - 0.5;
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()
    return x.size, x*scaleX, y*scaleY
def refinedHilbertCurve(order, cl_x, scale, plotOrNot):

    num,x,y = hilbertCurve(order,scale)
    ds = np.sqrt((y[1]-y[0])**2 + (x[1] - x[0])**2)
    n = max(np.ceil(ds/cl_x), 1) #n small segments on each segment
    nPoints = int((num-1)*n + 1)
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)
    k1 = 0
    k2 = 0
    for i in range(num-1):

        k2 = k1 + n

        xArray[k1:k2+1] = np.linspace(x[i],x[i+1],num=n+1)
        yArray[k1:k2+1] = np.linspace(y[i],y[i+1],num=n+1)

        k1 = k2

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray

# This is a S folding pattern
def sFolding(k,scaleX,scaleY):
    idx = np.zeros([4*k,4*k], np.int32)
    for i in range(k):
        idx[4*i  , 1:4*k] = range(i*(8*k), i*(8*k) + 4*k - 1)
        idx[4*i  , 0] = 4*k*(4*k-2*i)-1

        idx[4*i+1, 4*k-1] = i*(8*k) + 4*k-1
        idx[4*i+1, 0:4*k-1] = range(4*k*(4*k-2*i)-2, 4*k*(4*k-2*i) - 4*k-1, -1 )

        idx[4*i+2, 4*k-1] = i*(8*k) + 4*k
        idx[4*i+2, 0:4*k-1] = range(4*k*(4*k-2*i)-8*k+1, 4*k*(4*k-2*i) - 4*k)

        idx[4*i+3, 1:4*k] = range((i+1)*(8*k)-1,i*(8*k) + 4*k , -1)
        idx[4*i+3,0] = 4*k*(4*k-2*i)-8*k
    #idx = np.rot90(idx,-1)
    idx = np.fliplr(idx)
    y, x = np.indices(idx.shape).reshape(2, -1) /(4*k-1.0)
    x = x - 0.5;
    x[idx.ravel()], y[idx.ravel()] = x.copy(), y.copy()
    num= x.size
    x = x[4*k-2:]
    y =y[4*k-2:]
    return num-4*k+2, x*scaleX, y*scaleY

# This function refines a curve by inserting points based on a target segment length.
def curveRefine(num, xx, yy,cl_x, plotOrNot):
    numArray = np.zeros(num,np.int32)
    for i in range(num-1):
        ds = np.sqrt((yy[i+1]-yy[i])**2 + (xx[i+1] - xx[i])**2)
        n = max(np.ceil(ds/cl_x), 1) #n small segments on each segment
        numArray[i+1] = numArray[i] + n;

    nPoints = numArray[-1]+1
    xArray = np.zeros(nPoints)
    yArray = np.zeros(nPoints)

    for i in range(num-1):
        xArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(xx[i],xx[i+1],num=numArray[i+1] - numArray[i] + 1)
        yArray[numArray[i]:(numArray[i+1]+1)] = np.linspace(yy[i],yy[i+1],num=numArray[i+1] - numArray[i] + 1)

    if(plotOrNot):
        plt.plot(xArray, yArray,'-*')
        plt.ylim([-0.5,1.5])
        plt.xlim([-1,1])
        plt.show()
    return nPoints, xArray, yArray

def curveRefineUniform(xx, yy, cl_x, plotOrNot=True):
    """
    Densify a polyline with globally uniform spacing along arc length.

    Parameters:
    - xx, yy: control points of the original curve
    - cl_x: target segment length (uniform)
    - plotOrNot: whether to plot the result

    Returns:
    - nPoints: number of output points
    - xArray, yArray: uniformly spaced points along the curve
    """
    xx = np.array(xx)
    yy = np.array(yy)

    # Step 1: Compute segment lengths and cumulative arc length
    dx = np.diff(xx)
    dy = np.diff(yy)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    arc_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)

    total_length = arc_lengths[-1]
    nPoints = int(np.ceil(total_length / cl_x)) + 1
    target_distances = np.linspace(0, total_length, nPoints)

    # Step 2: Interpolate x and y using arc length parameterization
    xArray = np.interp(target_distances, arc_lengths, xx)
    yArray = np.interp(target_distances, arc_lengths, yy)

    # Step 3: Plot
    if plotOrNot:
        plt.plot(xx, yy, 'o--', label='Original')
        plt.plot(xArray, yArray, 'r-*', label='Uniform spacing')
        plt.title("Globally Uniform Curve Refinement")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    return nPoints, xArray, yArray

###########################################################################################################################
###########################################################################################################################

# Start to generate AEROS and Matcher input file

# num,x,y = candle(0.15, 0.157)
# num,x,y = hilbertCurve(2,0.32,0.32)
# num, x, y = sinCurve(False,xscale=0.17,yscale=0.22)
# num, x, y = sFolding(2,0.197,0.197)
nPoints, xArray, yArray = straightLine()
nPoints, xArray, yArray = curveRefineUniform(xArray,yArray, cl_x,True)

# Check length of curve
# length = np.sum(np.sqrt(np.diff(xArray)**2 + np.diff(yArray)**2))
# print('Length of the curve is %f' %length)


#############################################################################################################################
#For the canopy, it has nPoints-1 segments and is extruded 4 layers in the z direction,  the points are labeled as
#5 10 15 ... 5*nPoints
#4 9  14 ... 5*nPoints-1
#3 8  13 ... 5*nPoints-2
#2 7  12 ... 5*nPoints-3
#1 6  11 ... 5*nPoints-4
#For the cables, it has mPoints segments, 4 nodes connecting these cables are 1 , 4 , 4*nPoins - 3 , 4*nPoints, on the cables these nodes are
# 1 ,            5 ,            5*nPoins - 4 ,   5*nPoints,
# 5*nPoints + 1, 5*nPoints + 2, 5*nPoints + 3,   5*nPoints + 4,
# 5*nPoints + 5, 5*nPoints + 6, 5*nPoints + 7,   5*nPoints + 8,
# ......
# 5*nPoints + 4*mPoints - 7, 5*nPoints + 4*mPoints - 6, 5*nPoints + 4*mPoints - 5,   5*nPoints + 4*mPoints - 4,
# 5*nPoints + 4*mPoints - 3, 5*nPoints + 4*mPoints - 3, 5*nPoints + 4*mPoints - 3,   5*nPoints + 4*mPoints - 3,
# For the payload, it is a square panel
# 5*nPoints + 4*mPoints - 2, 5*nPoints + 4*mPoints - 1, 5*nPoints + 4*mPoints,   5*nPoints + 4*mPoints + 1
meshInclude = open('mesh.include','w')

# NodeId, x coordinate, y coordinate, z coordinate
meshInclude.write('NODES\n')
# First part for the canopy
# TAGG change
for i in range(nPoints):
    for j in range(nz):
        node_id = nz*i + j + 1
        z = j * cl_z
        meshInclude.write(f'{node_id}   {xArray[i]}  {yArray[i]}  {z} \n')

# Second part for 4 cables
cStart = np.array([
    [xArray[0], yArray[0], 0.0],
    [xArray[0], yArray[0], (nz - 1) * cl_z],
    [xArray[nPoints - 1], yArray[nPoints - 1], 0.0],
    [xArray[nPoints - 1], yArray[nPoints - 1], (nz - 1) * cl_z]
])
payloadPos = -3.0
cEnd = np.array([0.0, payloadPos, (nz // 2) * cl_z])

for i in range(mPoints - 1):
    # print('Adding cable points for segment %d of %d' % (i + 1, mPoints - 1))
    # i = 0, 1, 2 ..m-2
    for j in range(4):
        c = cStart[j, :] + (cEnd - cStart[j, :]) * (i + 1.0) / mPoints
        meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * i + j + 1, c[0], c[1], c[2]))
meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * mPoints - 3, cEnd[0], cEnd[1], cEnd[2]))

# Third part for payload
meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * mPoints - 2, -2 * cl_x * nPoints/20, payloadPos, cEnd[2] - (nz - 1) * cl_z/8))
meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * mPoints - 1, -2 * cl_x * nPoints/20, payloadPos, cEnd[2] + (nz - 1) * cl_z/8))
meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * mPoints,      2 * cl_x * nPoints/20, payloadPos, cEnd[2] - (nz - 1) * cl_z/8))
meshInclude.write('%d   %f  %f  %f \n' % (nz * nPoints + 4 * mPoints + 1,  2 * cl_x * nPoints/20, payloadPos, cEnd[2] + (nz - 1) * cl_z/8))
meshInclude.write('*\n')



# TopologyId, finite element type, node1Id, node2Id, node3Id ..
# (some element type has more nodes, but 129 is 3 nodes membrane element)
# First part for the canopy
topology = canopy_topology
meshInclude.write('TOPOLOGY\n')
elem_id = 1
diag_tracker = 0
flip_diagonal = False

for i in range(nPoints - 1):
    for j in range(nz - 1):
        n0 = nz * i + j + 1
        n1 = nz * (i + 1) + j + 1
        n2 = n1 + 1
        n3 = n0 + 1

        # Check every 10 quads, flip the diagonal
        if diag_tracker % num_flip_diag == 0:
            flip_diagonal = not flip_diagonal

        if flip_diagonal:
            # Normal diagonal
            meshInclude.write(f'{elem_id}   {topology}  {n0}  {n1} {n2} \n')
            elem_id += 1
            meshInclude.write(f'{elem_id}   {topology}  {n0}  {n2} {n3} \n')
            elem_id += 1
        else:
            # Flipped diagonal
            meshInclude.write(f'{elem_id}   {topology}  {n0}  {n1} {n3} \n')
            elem_id += 1
            meshInclude.write(f'{elem_id}   {topology}  {n1}  {n2} {n3} \n')
            elem_id += 1

        diag_tracker += 1

# Second part for 4 cables
# For the cables, segment ids are
# 8*nPoints-7,    8*nPoints-6,    8*nPoints-5,      8*nPoints-4
# 8*nPoints-3,    8*nPoints-2,    8*nPoints-1,      8*nPoints
# ......
# 8*nPoints + 4*mPoints-11,    8*nPoints + 4*mPoints - 10,    8*nPoints + 4*mPoints - 9,      8*nPoints + 4*mPoints - 8
cableTopology = 6
for i in range(mPoints):
    if(i == 0):
        a = [1, nz, nz*nPoints - (nz - 1), nz*nPoints]
    else:
        a = [nz*nPoints + 4*i - 3, nz*nPoints + 4*i - 2, nz*nPoints + 4*i - 1, nz*nPoints + 4*i]
    if(i == mPoints - 1):
        b = [nz*nPoints + 4*mPoints - 3] * 4
    else:
        b = [nz*nPoints + 4*i + 1, nz*nPoints + 4*i + 2, nz*nPoints + 4*i + 3, nz*nPoints + 4*i + 4]
    for k in range(4):
        meshInclude.write('%d   %d   %d   %d\n' % (elem_id, cableTopology, a[k], b[k]))
        elem_id += 1

# Third part for payload (4 elements)
# 8*nPoints + 4*mPoints-7,    8*nPoints + 4*mPoints-6,    8*nPoints + 4*mPoints-5,      8*nPoints + 4*mPoints-4
payloadTopology = canopy_topology
meshInclude.write('%d   %d  %d  %d %d \n' % (elem_id, payloadTopology,  nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints - 3)); elem_id += 1
meshInclude.write('%d   %d  %d  %d %d \n' % (elem_id, payloadTopology,  nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints - 3)); elem_id += 1
meshInclude.write('%d   %d  %d  %d %d \n' % (elem_id, payloadTopology,  nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3)); elem_id += 1
meshInclude.write('%d   %d  %d  %d %d \n' % (elem_id, payloadTopology,  nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3)); elem_id += 1

# elementId, element attribute (material) id
attributes = 1
meshInclude.write('ATTRIBUTES\n')
elem_id = 1
for i in range(nPoints - 1):
    for _ in range(2 * (nz - 1)):
        meshInclude.write('%d   %d \n' % (elem_id, attributes))
        elem_id += 1

cableAttributes = 2
for i in range(mPoints):
    for j in range(4):
        meshInclude.write('%d   %d \n' % (elem_id, cableAttributes))
        elem_id += 1

payloadAttributes = 1
for i in range(4):
    meshInclude.write('%d   %d \n' % (elem_id, payloadAttributes))
    elem_id += 1

tot_elements = elem_id - 1

meshInclude.write('*\n')
meshInclude.close()

###############################################################################################################################################################
parachuteInclude = open('parachute.include','w')
# Material specifies material
# material id, ....
youngsModulus = 6.08e8
poissonRatio = 0.4
density = 1153.4
thickness = 7.62e-5
parachuteInclude.write('MATERIAL\n')
# Canopy:
parachuteInclude.write('1 0 %f %f %f 0 0 %f 0 0 0 0 0 0 0\n' %(youngsModulus, poissonRatio, density, thickness))
# Cables:
parachuteInclude.write('2 2.0161242351057709e-05 1.820216e+10 4.0e-01 4.84660e+02 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.2e-11 3.1e-11 3.1e-11\n')
parachuteInclude.write('*\n')

# MATUSAGE specifies material
# start element number, end element number, material id
parachuteInclude.write('MATUSAGE\n')
parachuteInclude.write('1 %d 1\n' % (tot_elements - 8))
parachuteInclude.write('*\n')

# MATLAW can specify nonlinear property of the material
# material id, material name, ...
parachuteInclude.write('MATLAW\n')
parachuteInclude.write('1 HyperElasticPlaneStress %f %f %f %f\n' % (density, youngsModulus, poissonRatio, thickness))
parachuteInclude.write('*\n')

# TAGG REMOVED THIS
# # Pressure
# parachuteInclude.write('PRESSURE\n')
# parachuteInclude.write('1 %d %f\n' % (2 * (nz - 1) * (nPoints - 1), -4000.0))
# parachuteInclude.write('*\n')

# DISP for payload
parachuteInclude.write('DISP\n')
for freedom in range(1, 7):
    parachuteInclude.write('%d %d 0.0\n' % (nz*nPoints + 4*mPoints - 3, freedom))
    parachuteInclude.write('%d %d 0.0\n' % (nz*nPoints + 4*mPoints - 2, freedom))
    parachuteInclude.write('%d %d 0.0\n' % (nz*nPoints + 4*mPoints - 1, freedom))
    parachuteInclude.write('%d %d 0.0\n' % (nz*nPoints + 4*mPoints    , freedom))
    parachuteInclude.write('%d %d 0.0\n' % (nz*nPoints + 4*mPoints + 1, freedom))

# TAGG removed this
# # Symmetry constraints
# parachuteInclude.write('* symmetry planes\n')
# parachuteInclude.write('%d thru %d step %d 3 0.0\n' % (1, nz*nPoints - (nz - 1), nz))
# parachuteInclude.write('%d thru %d step %d 4 0.0\n' % (1, nz*nPoints - (nz - 1), nz))
# parachuteInclude.write('%d thru %d step %d 5 0.0\n' % (1, nz*nPoints - (nz - 1), nz))
# parachuteInclude.write('%d thru %d step %d 3 0.0\n' % (nz, nz*nPoints, nz))
# parachuteInclude.write('%d thru %d step %d 4 0.0\n' % (nz, nz*nPoints, nz))
# parachuteInclude.write('%d thru %d step %d 5 0.0\n' % (nz, nz*nPoints, nz))

parachuteInclude.close()

######################################################################################################################################################################
structureTop = open('structure.top','w')

structureTop.write('Nodes nodeset\n')
for i in range(nPoints):
    for j in range(nz):
        structureTop.write('%d   %f  %f  %f \n' % (nz*i + j + 1, xArray[i], yArray[i], j * cl_z))

# Second part for 4 cables
for i in range(mPoints - 1):
    for j in range(4):
        c = cStart[j,:] + (cEnd - cStart[j,:]) * (i + 1.0) / mPoints
        structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*i + j + 1, c[0], c[1], c[2]))
structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 3, cEnd[0], cEnd[1], cEnd[2]))

# Third part for payload
structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 2, -2*cl_x, payloadPos, cEnd[2] - (nz - 1) * cl_z/8))
structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 1, -2*cl_x, payloadPos, cEnd[2] + (nz - 1) * cl_z/8))
structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints    ,  2*cl_x, payloadPos, cEnd[2] - (nz - 1) * cl_z/8))
structureTop.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints + 1,  2*cl_x, payloadPos, cEnd[2] + (nz - 1) * cl_z/8))

topology = 4
structureTop.write('Elements StickMovingSurface using nodeset\n')
elem_id = 1
diag_tracker = 0
flip_diagonal = False

for i in range(nPoints - 1):
    for j in range(nz - 1):
        n0 = nz * i + j + 1
        n1 = nz * (i + 1) + j + 1
        n2 = n1 + 1
        n3 = n0 + 1

        # Check every 10 quads, flip the diagonal
        if diag_tracker % num_flip_diag == 0:
            flip_diagonal = not flip_diagonal

        if flip_diagonal:
            # Normal diagonal
            structureTop.write(f'{elem_id}   {topology}  {n0}  {n1} {n2} \n')
            elem_id += 1
            structureTop.write(f'{elem_id}   {topology}  {n0}  {n2} {n3} \n')
            elem_id += 1
        else:
            # Flipped diagonal
            structureTop.write(f'{elem_id}   {topology}  {n0}  {n1} {n3} \n')
            elem_id += 1
            structureTop.write(f'{elem_id}   {topology}  {n1}  {n2} {n3} \n')
            elem_id += 1

        diag_tracker += 1

structureTop.write('Elements cable using nodeset\n')
cableTopology = 1
for i in range(mPoints):
    if i == 0:
        a = [1, nz, nz*nPoints - (nz - 1), nz*nPoints]
    else:
        a = [nz*nPoints + 4*i - 3, nz*nPoints + 4*i - 2, nz*nPoints + 4*i - 1, nz*nPoints + 4*i]
    if i == mPoints - 1:
        b = [nz*nPoints + 4*mPoints - 3] * 4
    else:
        b = [nz*nPoints + 4*i + 1, nz*nPoints + 4*i + 2, nz*nPoints + 4*i + 3, nz*nPoints + 4*i + 4]
    for k in range(4):
        structureTop.write('%d   %d   %d   %d\n' % (2*(nz - 1)*(nPoints - 1) + 4*i - 7 + k + 1, cableTopology, a[k], b[k]))

structureTop.write('Elements payload using nodeset\n')
structureTop.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 7, topology, nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 6, topology, nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 5, topology, nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3))
structureTop.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 4, topology, nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3))

structureTop.close()

######################################################################################################################################################################

embeddedSurface = open('embeddedSurface.top','w')

embeddedSurface.write('Nodes nodeset\n')
for i in range(nPoints):
    for j in range(nz):
        embeddedSurface.write('%d   %f  %f  %f \n' % (nz*i + j + 1, xArray[i], yArray[i], j * cl_z))

# Second part for 4 cables
for i in range(mPoints - 1):
    for j in range(4):
        c = cStart[j,:] + (cEnd - cStart[j,:]) * (i + 1.0) / mPoints
        embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*i + j + 1, c[0], c[1], c[2]))
embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 3, cEnd[0], cEnd[1], cEnd[2]))

# Third part for payload
embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 2, -2*cl_x, payloadPos, 0.0))
embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints - 1, -2*cl_x, payloadPos, (nz - 1) * cl_z))
embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints    ,  2*cl_x, payloadPos, 0.0))
embeddedSurface.write('%d   %f  %f  %f \n' % (nz*nPoints + 4*mPoints + 1,  2*cl_x, payloadPos, (nz - 1) * cl_z))

topology = 4
embeddedSurface.write('Elements StickMovingSurface_8 using nodeset\n')
elem_id = 1
for i in range(nPoints - 1):
    for j in range(nz - 1):
        n0 = nz*i + j + 1
        n1 = nz*(i+1) + j + 1
        n2 = n1 + 1
        n3 = n0 + 1
        embeddedSurface.write('%d   %d  %d  %d %d \n' % (elem_id, topology, n0, n1, n2)); elem_id += 1
        embeddedSurface.write('%d   %d  %d  %d %d \n' % (elem_id, topology, n0, n2, n3)); elem_id += 1

embeddedSurface.write('Elements StickMovingSurface_9 using nodeset\n')
embeddedSurface.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 7, topology, nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 6, topology, nz*nPoints + 4*mPoints - 1, nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 5, topology, nz*nPoints + 4*mPoints + 1, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3))
embeddedSurface.write('%d   %d  %d  %d %d \n' % (8*nPoints + 4*mPoints - 4, topology, nz*nPoints + 4*mPoints - 2, nz*nPoints + 4*mPoints    , nz*nPoints + 4*mPoints - 3))

embeddedSurface.close()

######################################################################################################################################################################
surfaceTop = open('surface.top','w')
surfaceTop.write('SURFACETOPO 10 SURFACE_THICKNESS %f\n' % (thickness))
topology = 3
elem_id = 1
for i in range(nPoints - 1):
    for j in range(nz - 1):
        n0 = nz*i + j + 1
        n1 = nz*(i+1) + j + 1
        n2 = n1 + 1
        n3 = n0 + 1
        surfaceTop.write('%d   %d  %d  %d %d \n' % (elem_id, topology, n0, n1, n2)); elem_id += 1
        surfaceTop.write('%d   %d  %d  %d %d \n' % (elem_id, topology, n0, n2, n3)); elem_id += 1


surfaceTop.write('*\n')
surfaceTop.close()