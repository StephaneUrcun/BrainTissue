from __future__ import print_function
from dolfin import *
from ufl import eq
from ufl import nabla_div
import numpy as np

#########################
# Parameters
#########################

num_steps = 305
dT = Constant(0.001)
t = 0.0

#tissue specific
a = 800
poro_0 = 0.6
nu = 0.49
mu_l=35
mu_h=0.008

#sample 6N_3C_a
E=4330
k=2.9e-13
pl0=145

# Lame constants
lambda_ = (E*nu)/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))

############################
# Mesh, Boundaries and Space
############################

mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
	infile.read(mesh)

H=0.00345

x = SpatialCoordinate(mesh)
c = 0.0
r=abs(x[0])
	
left = CompiledSubDomain(" near(x[0],0.0) ", H=H)
right = CompiledSubDomain(" near(x[0],0.1*H)", H=H)
top =  CompiledSubDomain("near(x[1], H)",H=H)
bottom =  CompiledSubDomain("near(x[1], 0.0)")

boundary = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

boundary.set_all(0)
left.mark(boundary, 3)
right.mark(boundary, 4)
top.mark(boundary, 1)
bottom.mark(boundary, 2)

# Define Mixed Space (R3,[R2]^2) -> (u,pl,phl)
V = VectorElement("CG", mesh.ufl_cell(), 2)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
W1 = FunctionSpace(mesh, W)
W2 = FunctionSpace(mesh, W)
MS = FunctionSpace(mesh, MixedElement([V,W,W]))

# quadrature degree
q_degree = 6
dx = Measure('dx', domain = mesh)
dx = dx(metadata={'quadrature_degree': q_degree})


################################
# Constitutive relationships
################################

poro = Function(W1)
poro_n = Function(W1)
		
# Define strain
def epsilon(v):
	    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],[0, v[0]/r, 0],[v[1].dx(0),0, v[1].dx(1)]]))

# Vectorial div in cyl
def axi_div_2D(u):
	return u[0]/r+u[0].dx(0)+u[1].dx(1)

# Define sigma
def sigma_E(u):
	   return lambda_*tr(epsilon(u))*Identity(3) + 2.0*epsilon(u)

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
	cond=conditional(lt(pow(Sh(phl),2),1e-4),1e-4 ,pow(Sh(phl),2))
	return cond


# Define positivity of phl
def pos_pth(phl):
    cond = conditional(lt(phl,1e-2),0.0,phl)
    return cond

    
###########################
# Define boundary condition
###########################

bcu1  = DirichletBC(MS.sub(0).sub(0), 0.0 , left) # slip conditions
bcu2  = DirichletBC(MS.sub(0).sub(0), 0.0 , right) # 
bcu3  = DirichletBC(MS.sub(0).sub(1), 0.0 , bottom) 	# 
bcpl = DirichletBC(MS.sub(1), 0.0, top)  				# drained conditions
bcph = DirichletBC(MS.sub(2), 0.0, top)					#

bc=[bcu1,bcu2,bcu3,bcpl,bcph]
	
X0 = Function(MS)
Xn = Function(MS)
B = TestFunction(MS)
	
val_ps=Function(W2)
t_t_phl=Function(W2)

# Displacement field
e_u0 = Expression(('0.0', '0.0'), element=MS.sub(0).collapse().ufl_element())
e_pl0 =  Expression('8488.0', element=MS.sub(1).collapse().ufl_element() )#Pa=N/m^2 6N load
#e_pl0 =  Expression('4244.0', element=MS.sub(1).collapse().ufl_element() )#Pa=N/m^2 3N load

e_phl0=Expression('pl0',pl0=pl0, element=MS.sub(2).collapse().ufl_element() )


re_u0 = interpolate(e_u0, MS.sub(0).collapse())
pl0 = interpolate(e_pl0, MS.sub(1).collapse())
phl0 = interpolate(e_phl0, MS.sub(2).collapse())

assign(Xn, [re_u0, pl0, phl0])

(u,pl,phl)=split(X0)
(u_n,pl_n,phl_n)=split(Xn)
(v,ql,qhl)=split(B)


#########################
# Internal variable
#########################

# Define variation of porosity ECM
varporo_0= Expression('0.6', degree=1)
poro_n=project(varporo_0,W2)

def var_poro(u,u_n,poro_n):
	return (nabla_div(u-u_n)+poro_n)/(nabla_div(u-u_n)+1)
poro=project(var_poro(u,u_n,poro_n),W2)
	
#########################
# Weak formulation
#########################

#wetting phase
F = 2*pi*(1/dT)*Sl(phl)*nabla_div(u-u_n)*ql*r*dx + 2*pi*( 1/dT )*poro*dSldphl(phl)*(phl-phl_n)*ql*r*dx\
+ 2*pi*0.5*k_rl(phl)*(k/(mu_l))*dot(grad(pl),grad(ql))*r*dx\
+ 2*pi*0.5*k_rl(phl_n)*(k/(mu_l))*dot(grad(pl_n),grad(ql))*r*dx

#non-wetting phase
F += 2*pi*(1/dT)*Sh(phl)*nabla_div(u-u_n)*qhl*r*dx - 2*pi*( 1/dT )*poro*dSldphl(phl)*(phl-phl_n)*qhl*r*dx\
+ 2*pi*0.5*k_rh(phl)*(k/(mu_h))*dot(grad(pl+phl),grad(qhl))*r*dx\
+ 2*pi*0.5*k_rh(phl_n)*(k/(mu_h))*dot(grad(pl_n+phl_n),grad(qhl))*r*dx

#solid scaffold displacement
F += 2*pi*(1/dT)*((inner(2*mu*epsilon(u),epsilon(v))*r*dx + lambda_*axi_div_2D(u)*axi_div_2D(v)*r*dx\
- ps(pl,phl)*axi_div_2D(v)*r*dx)\
- (inner(2*mu*epsilon(u_n),epsilon(v))*r*dx + lambda_*axi_div_2D(u_n)*axi_div_2D(v)*r*dx\
- ps(pl_n,phl_n)*axi_div_2D(v)*r*dx))

#solver tuning
dX0 = TrialFunction(MS)
J = derivative(F, X0, dX0)
Problem = NonlinearVariationalProblem(F, X0, J = J, bcs = bc)
Solver  = NonlinearVariationalSolver(Problem)
Solver.parameters['newton_solver']['convergence_criterion'] = 'incremental'
Solver.parameters['newton_solver']['linear_solver'] = 'mumps'
Solver.parameters['newton_solver']['relative_tolerance'] = 1.e-15
Solver.parameters['newton_solver']['absolute_tolerance'] = 6.e-11
Solver.parameters['newton_solver']['maximum_iterations'] = 40


vtkfile_u = File('consolidation/u.pvd')
vtkfile_pl = File('consolidation/pl.pvd')
vtkfile_phl = File('consolidation/phl.pvd')

Sat_X=Function(MS)

print('dt=',float(dT))
for n in range(num_steps):
	print('n=',n)
	t += float(dT)
	print('t=',t)
	Solver.solve()
	(u,pl,phl)=X0.split()

	abs_phl=project(pos_pth(phl),W1)
	t_phl= interpolate(abs_phl, W2)
	assign(t_t_phl,t_phl)
	
	#update of porosity
	poro_n=project(poro,W1)
	poro=project(var_poro(u,u_n,poro_n),W1)

	if n == 19:
		dT.assign(0.01)
		print('dt=',float(dT))
	if n == 29:
		dT.assign(0.1)
		print('dt=',float(dT))
	if n == 39:
		dT.assign(1.0)
		print('dt=',float(dT))
	if n == 103:
		dT.assign(10.0)
		print('dt=',float(dT))
	if n == 127:
		dT.assign(30.0)
		print('dt=',float(dT))
	if n == 147:
		dT.assign(60.0)
		print('dt=',float(dT))
	if n == 176:
		dT.assign(150.0)
		print('dt=',float(dT))

	assign(Sat_X, [u, pl, t_t_phl])
	(_u,_pl,_phl)=Sat_X.split()

	if(n%10==0):
		vtkfile_u << (_u,t)
		vtkfile_pl << (_pl,t)
		vtkfile_phl << (_phl,t)
	
	assign(Xn, [u, pl,t_t_phl])
