from __future__ import print_function
from dolfin import *
from ufl import eq
from ufl import nabla_div
import numpy as np

#file = open("indent_1_5mm.txt","w")
file = open("indent_1mm.txt","w")
#file = open("indent_0_75mm.txt","w")

#########################
# Parameters
#########################

num_steps = 650
dT = Constant(0.5)
t = 0.0

#invariants:
a = 400
poro_0 = 0.5
nu = 0.47
mu_l=30
mu_h=0.003

#calibrated
E= 605
k= 2.8e-12
pl0=  17

# Lame constants
lambda_ = (E*nu)/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))


############################
# Mesh, Boundaries and Space
############################

mesh = Mesh()

with XDMFFile("mesh_cyl_2.xdmf") as infile:
	infile.read(mesh)

x = SpatialCoordinate(mesh)
r = abs(x[0])

bottom =  CompiledSubDomain("near(x[1], 0.0)")
right = CompiledSubDomain("near(x[0], 12e-3)")
left = CompiledSubDomain("near(x[0], 0.0)")

# Indenter diameter

#1.5mm
#top =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]>0.75e-3)")
#char =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]<=0.75e-3)")

#1mm
top =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]>=0.5e-3)")
char =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]<=0.50001e-3)")

#0.75mm
#top =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]>=0.375e-3)")
#char =  CompiledSubDomain("near(x[1], 5e-3) && (x[0]<=0.3750001e-3)")


boundary = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundary.set_all(0)

bottom.mark(boundary, 2)
left.mark(boundary, 3)
right.mark(boundary, 4)
top.mark(boundary, 1)
char.mark(boundary, 10)


# Define Mixed Space (R2,[R]^2) -> (u,pl,phl)
V = VectorElement("CG", mesh.ufl_cell(), 2)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
W1 = FunctionSpace(mesh, W)
V2 = FunctionSpace(mesh, V)
MS = FunctionSpace(mesh, MixedElement([V,W,W]))

# quadrature degree
q_degree = 6
dx = dx(metadata={'quadrature_degree': q_degree})
ds = Measure('ds', domain = mesh, subdomain_data = boundary)
N = Constant(('0.0','0.0','1.0'))

################################
# Constitutive relationships
################################

poro = Function(W1)
poro_n = Function(W1)

# Define strain
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],[0, v[0]/r, 0],[v[1].dx(0),0, v[1].dx(1)]]))

# Define stress
def sigma(u):
   return lambda_*tr(epsilon(u))*Identity(3) + 2.0*mu*epsilon(u)


#Define saturation and derivative
def Sh(phl):
	cond=conditional( lt(phl,0.0),0.0,(2/pi)*atan( phl/a ) )
	return cond

def Sl(phl):
	return 1-Sh(phl)

def dSldphl(phl):
	return -(2/((a)*pi))*( 1/( 1+pow((phl/(a)),2) ) )

def dShdphl(phl):
	return (2/((a)*pi))*( 1/( 1+pow((phl/(a)),2) ) )

# Define solid pressure
def ps(pl,phl):
    return pl+Sh(phl)*phl

# Define relative permeability
def k_rl(phl):
	cond=conditional(lt(pow(Sl(phl),1),1e-4),1e-4 ,pow(Sl(phl),1))
	return cond

def k_rh(phl):
	cond=conditional(lt(pow(Sh(phl),1),1e-4),1e-4 ,pow(Sh(phl),1))
	return cond


###################################################
# Prescribed displacement: Loading, Hold, Unloading
###################################################

expe_disp=np.zeros(num_steps)
for n in range(num_steps):

	if (n==0):
		expe_disp[n]=0.0
	if (n>0 and n<=90):
		expe_disp[n]=2.5e-6
	if (n>90 and n<=152):
		expe_disp[n]=1.25e-6
	if (n>152 and n<=192):
		expe_disp[n]=0.0
	if (n>192 and n<=204):
		expe_disp[n]=-2.5e-6
	if (n>204 and n<=219):
		expe_disp[n]=-1e-6
	if (n>219 and n<=500):
		expe_disp[n]=-0.5e-6
	if (n>500):
		expe_disp[n]=-1e-6


###########################
# Define boundary condition
###########################

bcux = DirichletBC(MS.sub(0).sub(1), -expe_disp[0] , char) 	# prescribed displacement
bcux2 = DirichletBC(MS.sub(0).sub(0), 0, char) 				# slip conditions
bcu1  = DirichletBC(MS.sub(0).sub(0), 0.0 , left) 			# 
bcu11  = DirichletBC(MS.sub(0).sub(0), 0.0 , right) 		# 
bcu2  = DirichletBC(MS.sub(0).sub(1), 0.0 , bottom) 		# 

bc=[bcu1,bcu2,bcu11,bcux,bcux2]


#########################
# Initial Conditions
#########################

# Define variational problem and initial condition
d_X0 = Function(MS)
Xn = Function(MS)
B = TestFunction(MS)

# Displacement field
e_u0 = Expression(('0.0', '0.0'), element=MS.sub(0).collapse().ufl_element())
e_pl0 =  Expression('0.0', element=MS.sub(1).collapse().ufl_element() )
e_phl0=Expression('pl0',pl0=pl0, element=MS.sub(2).collapse().ufl_element() )

re_u0 = interpolate(e_u0, MS.sub(0).collapse())
pl0 = interpolate(e_pl0, MS.sub(1).collapse())
phl0 = interpolate(e_phl0, MS.sub(2).collapse())

assign(Xn, [re_u0, pl0, phl0])

(d_u,d_pl,d_phl)=split(d_X0)
(u_n,pl_n,phl_n)=split(Xn)
(v,ql,qhl)=split(B)

#########################
# Internal variable
#########################

# Define variation of porosity ECM
varporo_0= Expression('poro',poro=poro_0, degree=1)
poro_n=project(varporo_0,W1)

def var_poro(d_u,poro_n):
	return (nabla_div(d_u)+poro_n)/(nabla_div(d_u)+1)
poro=project(var_poro(d_u,poro_n),W1)


#########################
# Weak formulation
#########################

#wetting phase
F = 2*pi*(1/dT)*Sl(pl_n+d_pl)*nabla_div(d_u)*ql*r*dx + 2*pi*( 1/dT )*poro*dSldphl(phl_n+d_phl)*(d_phl)*ql*r*dx\
+ 2*pi*k_rl(phl_n+d_phl)*(k/(mu_l))*dot(grad(pl_n+d_pl),grad(ql))*r*dx

#non-wetting phase
F += 2*pi*(1/dT)*Sh(phl_n+d_phl)*nabla_div(d_u)*qhl*r*dx - 2*pi*( 1/dT )*poro*dSldphl(phl_n+d_phl)*(d_phl)*qhl*r*dx\
+ 2*pi*k_rh(phl_n+d_phl)*(k/(mu_h))*dot(grad(pl_n+d_pl+phl_n+d_phl),grad(qhl))*r*dx

#solid scaffold displacement
F += 2*pi*(1/dT)*( (inner(sigma(u_n+d_u),epsilon(v))*r*dx - ps(pl_n+d_pl,phl_n+d_phl)*nabla_div(v)*r*dx)\
- (inner(sigma(u_n),epsilon(v))*r*dx - ps(pl_n,phl_n)*nabla_div(v)*r*dx))


#to represent saturations
sat_l=Function(W1)
sat_h=Function(W1)
sat_t=Function(W1)

Sat_X=Function(MS)
(sat_u,st_l,st_h)=split(Sat_X)

#solver tuning
ddX0 = TrialFunction(MS)
J = derivative(F, d_X0, ddX0)
Problem = NonlinearVariationalProblem(F, d_X0, J = J, bcs = bc)
Solver  = NonlinearVariationalSolver(Problem)
Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
Solver.parameters['newton_solver']['linear_solver'] = 'mumps'
Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-15
Solver.parameters['newton_solver']['absolute_tolerance'] = 6.e-11
Solver.parameters['newton_solver']['maximum_iterations'] = 35#30

vtkfile_u = File('indent_1mm/u.pvd')
vtkfile_pl = File('indent_1mm/pl.pvd')
vtkfile_phl = File('indent_1mm/phl.pvd')
vtkfile_poro = File('indent_1mm/poro.pvd')

t_du=Function(V2)
t_dpl=Function(W1)
t_dphl=Function(W1)
val_u=Function(V2)
val_pl=Function(W1)
val_phl=Function(W1)
val_u= interpolate(re_u0, V2)
val_pl= interpolate(e_pl0, W1)
val_phl= interpolate(e_phl0, W1)



print('dt=',float(dT))
for n in range(num_steps-1):
	print('n=',n)
	t += float(dT)
	print('t=',t)
	Solver.solve()
	(d_u,d_pl,d_phl)=d_X0.split()

	assign(t_du,d_u)
	assign(t_dpl,d_pl)
	assign(t_dphl,d_phl)

	val_u.vector().axpy(1, t_du.vector())    
	val_pl.vector().axpy(1, t_dpl.vector())   
	val_phl.vector().axpy(1, t_dphl.vector()) 

	# update of prescribed displacement
	bcux = DirichletBC(MS.sub(0).sub(1), -expe_disp[n+1] , char) 	

	bc=[bcu1,bcu11,bcu2,bcux,bcux2]

	Problem = NonlinearVariationalProblem(F, d_X0, J = J, bcs = bc)
	Solver  = NonlinearVariationalSolver(Problem)
	Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
	Solver.parameters['newton_solver']['linear_solver'] = 'mumps'
	Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-15
	Solver.parameters['newton_solver']['absolute_tolerance'] = 6.e-11
	Solver.parameters['newton_solver']['maximum_iterations'] = 35	

	if (n==90):
		dT.assign(0.25)
	if (n==192):
		dT.assign(0.5)
	if (n==204):
		dT.assign(0.2)
	if (n==219):
		dT.assign(0.1)
	if (n==500):
		dT.assign(0.2)
	
	#update of porosity
	poro_n=project(poro,W1)
	poro=project(var_poro(d_u,poro_n),W1)

	Traction=assemble(2*pi*dot(dot(sigma(val_u)- ps(val_pl,val_phl)*Identity(3), N),N)*r*ds(10))
	print("Traction (N)= ",float(Traction))
	
	assign(Sat_X, [val_u, val_phl, poro])
	(_sat_u,_val_phl,_poro)=Sat_X.split()

	if(n%10==0):
		vtkfile_u << (val_u,t)
		vtkfile_pl << (val_pl,t)
		vtkfile_phl << (val_phl,t)
		vtkfile_poro << (_poro,t)

	file.write("%.8f \t" % val_u(0.0,0.005)[1])
	file.write("%.8f \t" % -Traction)
	file.write("%.8f \t" % val_u(0.0025,0.005)[1])
	file.write("%.8f \t" % val_phl(0.0025,0.005) )
	file.write("%.6f \n" % t)

	assign(Xn, [val_u,val_pl,val_phl])

	X = SpatialCoordinate(mesh)
	x = X + d_u

file.close()


