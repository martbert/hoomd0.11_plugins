#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import _martini_plugin

import pair as custpair
import angle as custangle
import dihedral as custdihedral
from hoomd_script import *

class martini_nonbonded_params:
	# Initialize
	def __init__(self,fname=None):
		if fname == None:
			raise
		
		# Open file and read lines
		file = open(fname,'r')
		self.lines = file.readlines()
		self.n_lines = len(self.lines)
		
		file.close()
		
		# Create the atoms and nb_params objects
		self.atoms = dict()
		self.nb_params = dict()
		
		# Loop through the lines and call the appropriate subroutines
		# Define main modules
		modules = {'defaults':self.nothing, 'atomtypes':self.atomtypes, 'nonbond_params':self.nonbond_params}

		# If line is a module then perform action else continue reading file
		i = 0
		while i < self.n_lines:
			line = self.lines[i].strip('\n')
			if line.find('[')+1:
				# Replace all whitespace and strip ends
				module_name = line.replace(' ','').strip('[]')
				try:
					i = modules[module_name](i+1)
				except:
					pass
			i += 1

	def nothing(self,i):
		return i

	def atomtypes(self,i):
		# Loop through lines until '[' is found: skip empty lines and lines that start with ';'
		line = self.lines[i].strip('\n')
		while not line.find('[')+1:
			temp_line = line.replace(' ','')
			if not temp_line == '':
				if not temp_line[0] == ';':
					# Split line
					line = line.split()
					# Take first two entries and put them in type and mass respectively
					self.atoms[line[0]] = float(line[1])
			i += 1
			line = self.lines[i].strip('\n')

		return i-1
		
	def nonbond_params(self,i):
		# Loop through lines until '[' is found: skip empty lines and lines that start with ';'
		line = self.lines[i].strip('\n')
		while not line.find('[')+1:
			temp_line = line.replace(' ','')
			if not temp_line == '':
				if not temp_line[0] == ';':
					# Split line
					line = line.split()
					# Calculate sig and eps
					c6 = float(line[3])
					c12 = float(line[4])
					sig = (c12/c6)**(1./6.)
					eps = c6/4./sig**6.
					# Take first two entries and put them in type and mass respectively
					self.nb_params[(line[0],line[1])] = (sig,eps) 
			i += 1
			line = self.lines[i].strip('\n')

		return i-1

####################################################################
# Quick setup
####################################################################

def quick_setup(xml_file,itp_file):
	# Verify that the file is of the xml format if not return error
	index = xml_file.find('.xml')
	if index == -1:
		print "Error: quick_setup needs an xml file format"
		raise
	
	# Import atoms and atoms interactions
	params = martini_nonbonded_params(fname=itp_file)
	
	# Read system in
	system = init.read_xml(filename=xml_file)
	
	# Get all particle types
	types = []
	for p in system.particles:
		types.append(p.type)
	types = list(set(types))
	
	# Get all bonds type
	bonds = dict()
	for b in system.bonds:
		type = b.type
		# Get r0 and k
		r0 = float(type[type.find('r')+1:type.find('k')])
		k = float(type[type.find('k')+1:])
		bonds[type] = (k,r0)
		
	# Get all angles type
	pi = 3.14159265
	angles = dict()
	for a in system.angles:
		type = a.type
		# Get r0 and k
		p0 = float(type[type.find('p')+1:type.find('k')])/180.*pi
		k = float(type[type.find('k')+1:])
		#print "Angle of type %s has constant %f and eq. angle %f" % (type,k,p0)
		angles[type] = [k,p0]
	
	# Get all dihedrals type
	dihedrals = dict()
	for d in system.dihedrals:
		type = d.type
		# Get phi_0,k, and m
		k = float(type[type.find('k')+1:type.find('x')])
		phi0 = float(type[type.find('x')+1:type.find('n')])/180.*pi
		n = int(type[type.find('n')+1:])
		dihedrals[type] = [k,phi0,n]
		
	# Get all improper dihedrals type
	impropers = dict()
	for i in system.impropers:
		type = i.type
		# Get xi_0 and k
		k = 2.*float(type[type.find('k')+1:type.find('x')])
		xi0 = float(type[type.find('x')+1:])/180.*pi
		impropers[type] = [k,xi0]

	# Setup interactions
	#############################################################
	harm = None
	angle = None
	
	# Bonded interactions
	#####################
	if len(bonds) > 0:
		harm = bond.harmonic()
		for b in bonds:
			harm.set_coeff(b,k=bonds[b][0],r0=bonds[b][1])
	else:
		pass
	
	# Angles interactions
	#####################
	if len(angles) > 0:
		angle = custangle.cos2()
		for a in angles:
			angle.set_coeff(a,k=angles[a][0],t0=angles[a][1])
	else:
		pass

	# Dihedral interactions
	#####################
	if len(dihedrals) > 0:
		dih = custdihedral.harmonic()
		for d in dihedrals:
			print "Dihedral coeffs: k = %f, phi0 = %f, n = %d" % (dihedrals[d][0],dihedrals[d][1],dihedrals[d][2])
			dih.set_coeff(d,k=dihedrals[d][0],p0=dihedrals[d][1],n=dihedrals[d][2])
	else:
		pass	

	# Improper interactions
	#####################
	if len(impropers) > 0:
		imp = improper.harmonic()
		for i in impropers:
			imp.set_coeff(d,k=impropers[i][0],p0=impropers[i][1])
	else:
		pass	
	
	# Define all non-bonded interactions
	####################################
	# Coulomb parameters
	f = 138.935485
	er = 15.0
	
	# Base potentials
	pairlj = custpair.lj_martini()
	pairlj.set_params(mode='shift')
	paircoul = custpair.coulomb_martini()
	paircoul.set_params(mode='shift')
	
	# Loop over types to set all interactions
	for i in range(len(types)):
		for j in range(i,len(types)):
			type1 = types[i]
			type2 = types[j]
			# Get LJ parameters
			try:
				sig,eps = params.nb_params[(type1,type2)]
			except:
				try:
					sig,eps = params.nb_params[(type2,type1)]
				except:
					print "Could not find inter params between types %s and %s" % (type1,type2)
			#print type1,type2,sig,eps
			pairlj.pair_coeff.set(type1,type2,epsilon=eps,sigma=sig)
	paircoul.pair_coeff.set(types,types,f=f,er=er,ron=0.0)
	
	res = {"system":system, "pairlj":pairlj, "paircoul":paircoul}
	if len(bonds) > 0:
		res["bond"] = harm
	if len(angles) > 0:
		res["angle"] = angle
	if len(dihedrals) > 0:
		res["dihedral"] = dih
	if len(impropers) > 0:
		res["improper"] = imp

	return res