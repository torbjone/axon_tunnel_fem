TITLE Axonal voltage-gated potassium current
	doi:10.1038/nn.3132
	
	n8*h1 + n8*h2 Hodgkin-Huxley model
ENDCOMMENT
	gbar (pS/um2)
	temp = 33	(degC)		: original temp 
	tadj
	tadjh
	ik = ikv1

	tadj = q10^((celsius - temp)/10)
	tadjh = q10h^((celsius - temp)/10)

	nalpha = tadj*nalphafkt(v-vShift)