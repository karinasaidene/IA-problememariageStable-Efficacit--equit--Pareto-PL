import exemple # Pour pouvoir utiliser les methodes de exemple.py
from exemple import *
import copy
import random
import time
import matplotlib.pyplot as plt
import numpy as np

def lirePreferencesEtuSurSpe(nomFichier: str) -> list[list[int]]:
	lignes = lectureFichier(nomFichier)
	pref = []
	for i in range(1,len(lignes)):
		'''
		l=[]
		for j in range(2,len(lignes[i])):
			l.append(int(lignes[i][j]))
		'''
		l = [int(lignes[i][j]) for j in range(2,len(lignes[i]))]	#version liste compréhension
		pref.append(l)

	return pref

def lirePreferencesSpeSurEtu(nomFichier: str) -> tuple[list[list[int]],list[int]]:
	lignes = lectureFichier(nomFichier)
	pref = []
	cap = [int(x) for x in lignes[1][1:]]
	
	for i in range(2,len(lignes)):
		'''
		l=[]
		for j in range(2,len(lignes[i])):
			l.append(int(lignes[i][j]))
		'''
		l = [int(lignes[i][j]) for j in range(2,len(lignes[i]))]	#version liste compréhension
		pref.append(l)

	return cap,pref
	

def GaleShapleyCoteEtu(lPrefEtu: list[list[int]], capacites: list[int], lPrefSpe: list[list[int]]) -> list[tuple[int,int]]:
	"""
	en entrée :	la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chaque parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	en sortie :	un mariage parfait étudiant optimal M contenant des couples (étudiant, parcours)
	"""
	#initialisation
	prochaineProposition = [0 for i in range(len(lPrefEtu))]#indices des prochaines specialites auxquelles l'étudiant i n'a pas encore fait de proposition
	etusLibres = [i for i in range(len(lPrefEtu))]			#étudiants pas encore affectés (chacun d'un unique indice i dans [0,n-1])
	cap = copy.deepcopy(capacites)							#liste de places libre.  Il faut deepcopy pour que la liste en entrée ne soit pas modifiée
	M = []													#Mariage vide

	while len(etusLibres) > 0:
		e = etusLibres.pop()						#étudiant libre
		p = lPrefEtu[e][prochaineProposition[e]] 	#le parcours en question
		if (cap[p] > 0):							#si il y a une place libre dans ce parcours, affectation
			M.append((e,p))
			cap[p] -= 1
		else:							
			dejaAffectes = [etu for (etu,spe) in M if spe==p]		#liste d'étudiants déjà affectés à ce parcours
			permutation = False
			for etu in dejaAffectes:
				if lPrefSpe[p].index(e) < lPrefSpe[p].index(etu):	#si le parcours préfère e à un étudiant déjà affecté, permutation
					M.remove((etu,p))
					etusLibres.append(etu)
					M.append((e,p))
					permutation = True
					break
			if not permutation:			
				etusLibres.append(e)	#il faut remettre l'étudiant rejeté dans la liste

		prochaineProposition[e] += 1

	return M



def GaleShapleyCoteSpe(lPrefEtu: list[list[int]], capacites: list[int], lPrefSpe: list[list[int]]) -> list[tuple[int,int]]:	
	"""
	en entrée :	la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chaque parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	en sortie :	un mariage parfait parcours optimal M contenant des tuples (étudiant, parcours)   ...même ordre dans les tuples pour faciliter la comparaison des mariages plus tard
	"""
	#initialisation
	prochaineProposition = [0 for i in range(len(lPrefSpe))]#indices des prochains étudiants auxquels le parcours i n'a pas encore fait de proposition
	etusLibres = [i for i in range(len(lPrefEtu))]			#étudiants pas encore affectés (chacun d'un unique indice i dans [0,n-1])
	cap = copy.deepcopy(capacites)							#ainsi, aucune des trois listes ne sera modifiée
	M = []													#Mariage vide

	while any(cap):												#tant qu'il reste un parcours dont la capacité est non nulle
		p = cap.index(next(filter(lambda x : x>0, cap)))		#parcours avec au moins une place dispo
		e = lPrefSpe[p][prochaineProposition[p]]				#le prochain étudiant à recevoir une proposition
		if e in etusLibres:
			etusLibres.remove(e)
			M.append((e,p))
			cap[p] -= 1
		else:
			pCourant = next(filter(lambda paire : paire[0]==e, M))[1] #le parcours auquel l'étudiant e a déjà été affecté
			if lPrefEtu[e].index(p) < lPrefEtu[e].index(pCourant):
				M.remove((e,pCourant))
				cap[pCourant] += 1
				M.append((e,p))
				cap[p] -= 1
		prochaineProposition[p] += 1	

	return M

def pairesInstables(M: list[tuple[int,int]], lPrefEtu: list[list[int]], lPrefSpe: list[list[int]]) -> list[tuple[int,int]]:
	"""
	en entrée :	le mariage M représenté par des couples (étudiant, parcours)
				la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	en sortie :	une liste de paires instables représentée par des couples (étudiant, parcours)
	"""

	'''
	#version sans compréhension des listes
	paires = []
	for e1,p1 in M:
		for e2,p2 in M:
			if lPrefEtu[e1].index(p2) < lPrefEtu[e1].index(p1) and lPrefSpe[p2].index(e1) < lPrefSpe[p2].index(e2):
				paires.append((e1,p2))
	return paires
	'''
	#version avec compréhension des listes
	return [(e1,p2) for e1,p1 in M for e2,p2 in M if lPrefEtu[e1].index(p2) < lPrefEtu[e1].index(p1) and lPrefSpe[p2].index(e1) < lPrefSpe[p2].index(e2)]



def genererPrefEtu(n):	
	lPrefEtu=[]
	for i in range(n):
		prefEtu = [i for i in range(9)] #9 parcours
		random.shuffle(prefEtu)
		lPrefEtu.append(prefEtu)
	return lPrefEtu

def genererPrefSpe(n):	
	lPrefSpe=[]
	for i in range(9):
		prefSpe = [i for i in range(n)]
		random.shuffle(prefSpe)
		lPrefSpe.append(prefSpe)
	return lPrefSpe

def genererCap(n):
	#la somme des capacites est de n du faite que le reste est rajoyte a la fins
	cap = [n//9 for i in range(9)]
	reste = n % 9 
	for i in range(reste):
		cap[i] += 1
	return cap


def genererPref(n: int) -> tuple[list[list[int]], list[int], list[list[int]]]:
	"""
	en entrée:	le nombre d'étudiants n
				(le nombre de parcours est constante: 9)
	en sortie:	
				la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chacun des 9 parcours, générés de façon détérministe et plus ou moins égaux entre les parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	"""
	return genererPrefEtu(n), genererCap(n), genererPrefSpe(n)



def temps_execution_gs_cote_etu(r:range, nb_tests:int) -> list[float]:
	assert(nb_tests > 0)
	temps = []
	for i in r:
		total=0
		for j in range(nb_tests):
			lpe,c,lps = genererPref(i)
			tps1 = time.process_time()
			GaleShapleyCoteEtu(lpe,c,lps)
			tps2 = time.process_time()
			total += (tps2 - tps1)
		temps.append(total/nb_tests)
	return temps

def temps_execution_gs_cote_spe(r:range, nb_tests:int) -> list[float]:
	assert(nb_tests > 0)
	temps = []
	for i in r:
		total=0
		for j in range(nb_tests):
			lpe,c,lps = genererPref(i)
			tps1 = time.process_time()
			GaleShapleyCoteSpe(lpe,c,lps)
			tps2 = time.process_time()
			total += (tps2-tps1)
		temps.append(total/nb_tests)
	return temps

def nb_iter_gs_cote_etu(r:range, nb_tests:int) -> list[float]:
	assert(nb_tests > 0)
	nb_iter = []
	for i in r:
		total=0
		for j in range(nb_tests):
			#GaleShapleyCoteEtu_nbIter(genererPref[i])[1] represente it retourné par la fonction GS
			#M,it = GaleShapleyCoteEtu_nbIter(genererPref(i))
			lPrefEtu, cap, lPrefSpe = genererPref(i)
			total += GaleShapleyCoteEtu_nbIter(lPrefEtu, cap, lPrefSpe)[1]
		nb_iter.append(total/nb_tests)
	return nb_iter

def nb_iter_gs_cote_spe(r:range, nb_tests:int) -> list[float]:
	assert(nb_tests > 0)
	nb_iter = []
	for i in r:
		total=0
		for j in range(nb_tests):
			#GaleShapleyCoteEtu_nbIter(genererPref[i])[1] represente it retourné par la fonction GS
			#M,it = GaleShapleyCoteSpe_nbIter(genererPref(i))
			lPrefEtu, cap, lPrefSpe = genererPref(i)
			total += GaleShapleyCoteSpe_nbIter(lPrefEtu, cap, lPrefSpe)[1]
		nb_iter.append(total/nb_tests)
	return nb_iter




def GaleShapleyCoteEtu_nbIter(lPrefEtu: list[list[int]], capacites: list[int], lPrefSpe: list[list[int]]) -> list[tuple[int,int]]:
	'''
	en entrée :	la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chaque parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	en sortie :	un mariage parfait étudiant optimal M contenant des couples (étudiant, parcours)
	'''
	#initialisation
	prochaineProposition = [0 for i in range(len(lPrefEtu))]#indices des prochaines specialites auxquelles l'étudiant i n'a pas encore fait de proposition
	etusLibres = [i for i in range(len(lPrefEtu))]			#étudiants pas encore affectés (chacun d'un unique indice i dans [0,n-1])
	cap = copy.deepcopy(capacites)							#liste de places libre.  Il faut deepcopy pour que la liste en entrée ne soit pas modifiée
	M = []													#Mariage vide

	it = 0
	while len(etusLibres) > 0:
		e = etusLibres.pop()						#étudiant libre
		p = lPrefEtu[e][prochaineProposition[e]] 	#le parcours en question
		if (cap[p] > 0):							#si il y a une place libre dans ce parcours, affectation
			M.append((e,p))
			cap[p] -= 1
		else:							
			dejaAffectes = [etu for (etu,spe) in M if spe==p]		#liste d'étudiants déjà affectés à ce parcours
			permutation = False
			for etu in dejaAffectes:
				if lPrefSpe[p].index(e) < lPrefSpe[p].index(etu):	#si le parcours préfère e à un étudiant déjà affecté, permutation
					M.remove((etu,p))
					etusLibres.append(etu)
					M.append((e,p))
					permutation = True
					break
			if not permutation:			
				etusLibres.append(e)	#il faut remettre l'étudiant rejeté dans la liste
		prochaineProposition[e] += 1
		it += 1

	return M, it



def GaleShapleyCoteSpe_nbIter(lPrefEtu: list[list[int]], capacites: list[int], lPrefSpe: list[list[int]]) -> list[tuple[int,int]]:	
	'''
	en entrée :	la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chaque parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
	en sortie :	un mariage parfait parcours optimal M contenant des tuples (étudiant, parcours)   ...même ordre dans les tuples pour faciliter la comparaison des mariages plus tard
	'''
	#initialisation
	prochaineProposition = [0 for i in range(len(lPrefSpe))]#indices des prochains étudiants auxquels le parcours i n'a pas encore fait de proposition
	etusLibres = [i for i in range(len(lPrefEtu))]			#étudiants pas encore affectés (chacun d'un unique indice i dans [0,n-1])
	cap = copy.deepcopy(capacites)							#ainsi, aucune des trois listes ne sera modifiée
	M = []													#Mariage vide

	it = 0
	while any(cap):												#tant qu'il reste un parcours dont la capacité est non nulle
		p = cap.index(next(filter(lambda x : x>0, cap)))		#parcours avec au moins une place dispo
		e = lPrefSpe[p][prochaineProposition[p]]				#le prochain étudiant à recevoir une proposition
		if e in etusLibres:
			etusLibres.remove(e)
			M.append((e,p))
			cap[p] -= 1
		else:
			pCourant = next(filter(lambda paire : paire[0]==e, M))[1] #le parcours auquel l'étudiant e a déjà été affecté
			if lPrefEtu[e].index(p) < lPrefEtu[e].index(pCourant):
				M.remove((e,pCourant))
				cap[pCourant] += 1
				M.append((e,p))
				cap[p] -= 1
		prochaineProposition[p] += 1	
		it += 1

	return M, it




def genererPL(lPrefEtu: list[list[int]], capacites: list[int], lPrefSpe: list[list[int]], k: int):
	"""
	en entrée :	la matrice lPrefEtu[i,j] des préférences j pour chaque étudiant i
				la liste d'entiers des capacités de chaque parcours
				la matrice lPrefSpe[i,j] des préférences j pour chaque parcours i
				l'entier k : ce programme écrit un PL (par effet de bord) pour le nombre d'arêtes d'un graphe maxima
	"""
	n = len(lPrefEtu)	#n=nb étudiants
	m = len(lPrefSpe)	#m=nb étudiants
	
	kPremPrefEtu = [ligne[:k] for ligne in lPrefEtu]
	kPremPrefSpe = lPrefSpe#[ligne[:k] for ligne in lPrefSpe]
	E=[]
	for i,ligne in enumerate(kPremPrefEtu):
		for spe in ligne:
			if i in kPremPrefSpe[spe]:
				E.append((i,spe))

	#for i in range(len(lPrefEtu)):
		#for couple in []

	f = open("Q9.lp","w+")
	f.write("Maximize\n")
	f.write("obj: ")
	contraintes_etu = np.zeros((n,len(E)))	#matrice[etudiant i, arete i]
	contraintes_spe = np.zeros((m,len(E)))	#matrice[specialisation i, arete j]
	for i,couple in enumerate(E):
		etu,spe = couple
		f.write("x"+str(i))		#écriture de l'objective
		if i < len(E)-1:
			f.write(" + ")
		
		assert(contraintes_etu[etu][i] == 0)	#préparation des matrices pour écriture des contraintes
		contraintes_etu[etu][i] = 1

		assert(contraintes_spe[spe][i] == 0)
		contraintes_spe[spe][i] = 1


	#écriture des contraintes
	f.write("\nSubject To\n")
	p=1									#indice p de la contrainte...les autres lettres étant prises
	for etu_list in contraintes_etu:
		f.write("c"+str(p)+": ")
		for i in range(len(etu_list)):	#xi est la variable en question
			if i==1:
				f.write("x"+str(i))
				if i < (len(etu_list)-1):
					f.write(" + ")
		f.write(" <= 1\n")
		p+=1

	for j, spe_list in enumerate(contraintes_spe):
		f.write("c"+str(p)+": ")
		for i in range(len(spe_list)):
			if i==1:
				f.write("x"+str(i))
			if i < (len(etu_list)-1):
				f.write(" + ")
		f.write(" <= "+str(capacites[j])+"\n")
		p+=1

	f.write("Binary\n")
	for i in range(len(E)):
		f.write("x"+str(i)+" ")
	f.write("\nEnd\n")
	f.close()











	
	



'''
print("bonjour")
maListe=exemple.lectureFichier("test.txt") # Execution de la methode lectureFichier du fichier exemple.
print(maListe)
exemple.createFichierLP(maListe[0][0],int(maListe[1][0])) #Methode int(): transforme la chaine de caracteres en entier
'''

def main():
#Q1
	etuSurSpe = lirePreferencesEtuSurSpe("PrefEtu.txt")
	cap, speSurEtu = lirePreferencesSpeSurEtu("PrefSpe.txt")	#couple (liste des capacités, matrice des préférences)

	###TESTS
	#print('etuSurSpe:')
	#print(etuSurSpe)
	#print()
	#print('cap')
	#print(cap)
	#print()
	#print('speSurEtu:')
	#print(speSurEtu)
	#print()
	assert(len(cap) == len(speSurEtu))	

#Q2
	MEtuOpt = GaleShapleyCoteEtu(etuSurSpe, cap, speSurEtu)
	print("Mariage parfait côté étudiant : couples de (étudiant, parcours)")
	print(MEtuOpt)	
	print()

#Q3
	MSpeOpt = GaleShapleyCoteSpe(etuSurSpe, cap, speSurEtu)
	print("Mariage parfait côté parcours : couples de (étudiant, parcours)")
	print(MSpeOpt)
	print()

#Q4
	instablesMEtuOpt = pairesInstables(MEtuOpt, etuSurSpe, speSurEtu)
	print("Paires instables du mariage parfait côté étudiant")
	print(instablesMEtuOpt)
	print()

	instablesMSpeOpt = pairesInstables(MSpeOpt, etuSurSpe, speSurEtu)
	print("Paires instables du mariage parfait côté parcours")
	print(instablesMSpeOpt)
	print()

#Q5
	#cf genererPref qui appelle les fonctions

#Q6

#COMPLEXITE TEMPORELLE - GRAPHES
	'''
	#on crée deux graphes:
	fig, axes = plt.subplots(nrows=2, ncols=1) 


	#Graphe des deux courbes ensemble:
	abscisse = range(200,2000,200)
	temps_cote_etu = temps_execution_gs_cote_etu(abscisse,nb_tests=10)
	temps_cote_spe = temps_execution_gs_cote_spe(abscisse,nb_tests=10)
	
	axes[0].plot(abscisse, temps_cote_etu, color = "blue", label="côté étudiant")
	axes[0].plot(abscisse, temps_cote_spe, color = "green", label="côté spécialité")
	axes[0].legend()
	axes[0].set_title("Temps d'exécution de GS coté étudiant, coté spécialité")
	axes[0].set_xlabel("n")
	axes[0].set_ylabel("temps (s)")

	#Graphe de la courbe côté étudiant pour voir plus clairement l'évolution de son taux de croissance
	abscisse = range(500,5000,500)
	temps_cote_etu = temps_execution_gs_cote_etu(abscisse, nb_tests=10)

	axes[1].plot(abscisse, temps_cote_etu, color = "blue", label="côté étudiant")
	axes[1].legend()
	axes[1].set_title("Temps d'exécution de GS côté étudiant")
	axes[1].set_xlabel("n")
	axes[1].set_ylabel("temps (s)")	

	plt.tight_layout()
	#plt.show()


#Q7
	#Oui, la complexité est polynomiale de degré strictement supérieur à 1.  C'est cohérent avec la complexité théorique de O(n**2)

#Q8

	#NOMBRE MOYEN D'ITERATIONS - GRAPHES
	
	#Un seul graphe suffit
	fig2,axes2 = plt.subplots()

	abscisse = range(200,2000,200)
	it_cote_etu = nb_iter_gs_cote_etu(abscisse, 10)
	it_cote_spe = nb_iter_gs_cote_spe(abscisse, 10)
	axes2.plot(abscisse, it_cote_etu, color='blue', label='nb itérations côté étudiant')
	axes2.plot(abscisse, it_cote_spe, color='green', label='nb itérations côté spécialité')
	axes2.legend()
	axes2.set_title("Nombre d'itérations de GS côté étudiant, côté spécialité")
	axes2.set_xlabel("n")
	axes2.set_ylabel("nombre d'itérations")
	plt.show()

	#On voit que le nombre d'itérations est linéaire par rapport à n.  C'est cohérent avec la complexité théorique, i.e. O(n) tours de boucles, chacun en O(n) pour une complexité globale de O(n**2)
	'''

#Q9
	prefEtu, cap, prefSpe = genererPref(15)
	genererPL(prefEtu, cap, prefSpe, 1)


if __name__ == "__main__":
	main()


