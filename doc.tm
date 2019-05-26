<TeXmacs|1.99.8>

<style|generic>

<\body>
  <subsection|Computation of trajectories>

  In order to compute the particle trajectories, we exploit the lagrangian
  structure of the dynamics.

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<cal-L\>>|<cell|=>|<cell|e*<around*|(|u*<wide|B|\<vect\>>+<wide|A|\<vect\>>|)>\<cdot\><wide|<wide|X|\<vect\>>|\<dot\>>-H<around*|(|<wide|X|\<vect\>>,u|)>>>|<row|<cell|H>|<cell|=>|<cell|<frac|e<rsup|2>|2*m>*u<rsup|2>*B<rsup|2><around*|(|\<psi\>,\<theta\>|)>+\<mu\>*B<around*|(|\<psi\>,\<theta\>|)>+e*\<Phi\><around*|(|\<psi\>,\<theta\>|)>>>|<row|<cell|<wide|A|\<vect\>>>|<cell|=>|<cell|-\<psi\>*<wide|\<nabla\>|\<vect\>>\<varphi\>+\<psi\><rsub|pol>*<wide|\<nabla\>|\<vect\>>\<theta\>>>|<row|<cell|<wide|B|\<vect\>>>|<cell|=>|<cell|<wide|\<nabla\>|\<vect\>>\<times\><wide|A|\<vect\>>>>>>
  </eqnarray*>

  with <math|<wide|X|\<vect\>>> the particle position, and <math|u=m*v/e*B>
  the parallel Larmor radius. The particle position is parametered by
  <math|\<psi\>=A<rsub|\<varphi\>>> the magnetic flux, <math|\<theta\>> and
  <math|\<varphi\>> the poloidal and toroidal angles. The toroidal symmetry
  allows to write the conservation of the toroidal angular momentum

  <\eqnarray*>
    <tformat|<table|<row|<cell|L>|<cell|=>|<cell|u*B<rsub|\<varphi\>>+\<psi\>>>>>
  </eqnarray*>

  Then, the lagrangian can be written in coordinates
  <math|\<psi\>,\<theta\>,\<varphi\>,L>.

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<cal-L\><around*|[|\<psi\><around*|(|t|)>,\<theta\><around*|(|t|)>,\<varphi\><around*|(|t|)>;L|]>>|<cell|=>|<cell|e*<around*|(|<frac|B<rsub|\<theta\>>|B<rsub|\<varphi\>>>*<around*|(|L-\<psi\>|)>+A<rsub|\<theta\>>|)>*<wide|\<theta\>|\<dot\>>+e*L*<wide|\<varphi\>|\<dot\>>-H<around*|(|\<psi\>,\<theta\>,L|)>>>|<row|<cell|H<around*|(|\<psi\>,\<theta\>;L|)>>|<cell|=>|<cell|<frac|e<rsup|2>|2*m>*<around*|(|L-\<psi\>|)><rsup|2>*<frac|B<rsup|2><around*|(|\<psi\>,\<theta\>|)>|B<rsub|\<varphi\>><rsup|2><around*|(|\<psi\>,\<theta\>|)>>+\<mu\>*B<around*|(|\<psi\>,\<theta\>|)>+e*\<Phi\><around*|(|\<psi\>,\<theta\>|)>>>>>
  </eqnarray*>

  The trajectory can then be computed by seeking level lines of the
  hamiltonian as

  <\eqnarray*>
    <tformat|<table|<row|<cell|H<around*|(|\<psi\><around*|(|\<theta\>;L|)>,\<theta\>;L|)>>|<cell|=>|<cell|E>>>>
  </eqnarray*>

  and the banana tips can be defined as the solutions of the equation system

  <\eqnarray*>
    <tformat|<table|<row|<cell|H<around*|(|\<psi\><rsub|b>,\<theta\><rsub|b>;L|)>>|<cell|=>|<cell|E>>|<row|<cell|<frac|\<partial\>H|\<partial\>\<psi\>><around*|(|\<psi\><rsub|b>,\<theta\><rsub|b>;L|)>>|<cell|=>|<cell|0>>>>
  </eqnarray*>

  The temporal and toroidal displacements can be defined by computing the
  variations of the Lagrangian with respect to <math|\<psi\>> and <math|L>

  <\eqnarray*>
    <tformat|<table|<row|<cell|L<rsub|\<theta\>>>|<cell|=>|<cell|<frac|B<rsub|\<theta\>>|B<rsub|\<varphi\>>>*<around*|(|L-\<psi\>|)>+A<rsub|\<theta\>>>>|<row|<cell|<frac|\<mathd\>t|\<mathd\>\<theta\>>>|<cell|=>|<cell|<frac|\<partial\><rsub|\<psi\>>L<rsub|\<theta\>>|\<partial\><rsub|\<psi\>>H>>>|<row|<cell|<frac|\<mathd\>\<varphi\>|\<mathd\>\<theta\>>>|<cell|=>|<cell|\<partial\><rsub|L>H*<frac|\<mathd\>t|\<mathd\>\<theta\>>-\<partial\><rsub|L>L<rsub|\<theta\>>>>>>
  </eqnarray*>

  In the special case of the trapped particles, the denominator
  <math|\<partial\><rsub|\<psi\>>H> vanishes as
  <math|<sqrt|\<theta\><rsub|b>-\<theta\>>>, so we can regularise the
  integration so the last point is weighted as

  <\equation*>
    <big|int><rsub|<around*|\<lfloor\>|\<theta\><rsub|b>|\<rfloor\>>><rsup|\<theta\><rsub|b>><frac|\<mathd\>\<theta\>|<sqrt|\<theta\><rsub|b>-\<theta\>>>=2*<around*|(|\<theta\><rsub|b>-<around*|\<lfloor\>|\<theta\><rsub|b>|\<rfloor\>>|)>
  </equation*>

  where <math|<around*|\<lfloor\>|\<theta\><rsub|b>|\<rfloor\>>> denotes the
  preceding <math|\<theta\>> grid node.<em|>

  <subsection|Computation of the kernel integral>

  Perturbed Vlasov equation around an equilibrium <math|F<rsub|eq>>

  <\equation*>
    \<partial\><rsub|t>f+<around*|{|H<rsub|eq>,f|}>=<around*|{|F<rsub|eq>,h|}>
  </equation*>

  The solution of this equation is given by the Duhamel integral

  <\equation*>
    f<around*|(|0,\<psi\>,\<theta\>,\<varphi\>,v<rsub|<around*|\|||\|>>,\<mu\>|)>=<big|int><rsub|-\<infty\>><rsup|0>\<mathd\>t*<around*|{|F<rsub|eq>,h|}><around*|(|t,\<psi\><around*|(|t|)>,\<theta\><around*|(|t|)>,\<varphi\><around*|(|t|)>,v<rsub|<around*|\|||\|>><around*|(|t|)>,\<mu\>|)>
  </equation*>

  with <math|\<psi\><around*|(|t|)>> the backwards trajectory ending at
  <math|\<psi\>>, and where we have used <math|<wide|\<mu\>|\<dot\>>=0>. The
  variational form of the interaction writes

  <\equation*>
    S<rsub|int>=-<frac|1|2>*<big|int>h<rsup|\<dag\>><around*|(|\<psi\>,\<theta\>,\<varphi\>,v<rsub|<around*|\|||\|>>,\<mu\>|)>*<big|int><rsub|-\<infty\>><rsup|0>\<mathd\>t*<around*|{|F<rsub|eq>,h|}><around*|(|t,\<psi\><around*|(|t|)>,\<theta\><around*|(|t|)>,\<varphi\><around*|(|t|)>,v<rsub|<around*|\|||\|>><around*|(|t|)>,\<mu\>|)>
  </equation*>

  In order to express that, we do a temporal and toroidal Fourier transform
  on the hamiltonian. This gives

  <\eqnarray*>
    <tformat|<table|<row|<cell|S<rsub|int><rsup|n*\<omega\>>>|<cell|=>|<cell|-<frac|1|2>*<big|int>h<rsup|\<dag\>><around*|(|\<psi\>,\<theta\>,v<rsub|<around*|\|||\|>>,\<mu\>|)>*<big|int><rsub|-\<infty\>><rsup|0>\<mathd\>t*<around*|{|F<rsub|eq>,h|}><around*|(|\<psi\><around*|(|t|)>,\<theta\><around*|(|t|)>,v<rsub|<around*|\|||\|>><around*|(|t|)>,\<mu\>|)>>>|<row|<cell|>|<cell|>|<cell|exp<around*|[|-i*\<omega\>*t+i*n*<around*|(|\<varphi\><around*|(|t|)>-\<varphi\><around*|(|0|)>|)>|]>>>>>
  </eqnarray*>

  <subsubsection|Case for passing particles>

  We change the variable in the integral to parametrize in angle
  <math|\<theta\><rprime|'>>, and use the stationary trajectory information
  depending on <math|\<theta\><rprime|'>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|S<rsub|int><rsup|n*\<omega\>>>|<cell|=>|<cell|-<frac|1|2>*<big|int>F<rsub|eq><around*|(|\<mathd\>P<rsub|\<varphi\>>*\<mathd\>E*\<mathd\>\<mu\>|)>*<big|oint><frac|\<mathd\>\<theta\>|2*\<pi\>>*h<rsup|\<dag\>><around*|(|\<psi\><around*|(|\<theta\>|)>,\<theta\>,v<rsub|<around*|\|||\|>><around*|(|\<theta\>|)>,\<mu\>|)>>>|<row|<cell|>|<cell|>|<cell|<big|int><rsub|-\<infty\>><rsup|+\<infty\>><frac|\<mathd\>\<theta\><rprime|'>|<wide|\<theta\>|\<dot\>><rprime|'>>*<around*|{|ln*F<rsub|eq>,h|}><around*|(|\<psi\><around*|(|\<theta\><rprime|'>|)>,\<theta\><rprime|'>,v<rsub|<around*|\|||\|>><around*|(|\<theta\><rprime|'>|)>,\<mu\>|)>>>|<row|<cell|>|<cell|>|<cell|exp<around*|[|-i*\<omega\>*<around*|(|t<around*|(|\<theta\><rprime|'>|)>-t<around*|(|\<theta\>|)>|)>+i*n*<around*|(|\<varphi\><around*|(|\<theta\><rprime|'>|)>-\<varphi\><around*|(|\<theta\>|)>|)>|]>*\<bbb-1\><around*|[|t<around*|(|\<theta\><rprime|'>|)>\<leqslant\>t<around*|(|\<theta\>|)>|]>>>>>
  </eqnarray*>

  The infinite integral on <math|\<theta\><rprime|'>> corresponds a simple
  change of variable with respect to time. In order to account for the
  periodicity of the trajectories, we need to fold this integral into an
  integral on a single period. We can integrate it out explicitly using the
  bounce quantities

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<tau\><rsub|b>>|<cell|=>|<cell|t<around*|(|\<theta\>\<pm\>2*\<pi\>|)>-t<around*|(|\<theta\>|)>>>|<row|<cell|\<varphi\><rsub|b>>|<cell|=>|<cell|\<varphi\><around*|(|\<theta\>\<pm\>2*\<pi\>|)>-\<varphi\><around*|(|\<theta\>|)>>>>>
  </eqnarray*>

  where the sign <math|\<pm\>> is chosen to have positive
  <math|\<tau\><rsub|b>>. Then, we can perform the periodicity sum
  analytically as

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|sum><rsub|K\<geqslant\>0>*exp<around*|[|i*\<omega\>*K*\<tau\><rsub|b>-i*n*K*\<varphi\><rsub|b>|]>>|<cell|=>|<cell|<frac|1|1-exp<around*|{|i*<around*|[|\<omega\>*\<tau\><rsub|b>-n*\<varphi\><rsub|b>|]>|}>>>>>>
  </eqnarray*>

  and are left with a <math|\<theta\><rprime|'>> integral spanning the first
  past trajectory.

  <\eqnarray*>
    <tformat|<table|<row|<cell|S<rsub|int><rsup|n*\<omega\>>>|<cell|=>|<cell|-<frac|1|2>*<big|int><frac|F<rsub|eq><around*|(|\<mathd\>P<rsub|\<varphi\>>*\<mathd\>E*\<mathd\>\<mu\>|)>|1-exp<around*|{|i*<around*|[|\<omega\>*\<tau\><rsub|b>-n*\<varphi\><rsub|b>|]>|}>>>>|<row|<cell|>|<cell|>|<cell|<big|oint><frac|\<mathd\>\<theta\>|2*\<pi\>>*h<rsup|\<dag\>><around*|(|\<psi\><around*|(|\<theta\>|)>,\<theta\>,v<rsub|<around*|\|||\|>><around*|(|\<theta\>|)>,\<mu\>|)>*exp<around*|{|i*<around*|[|\<omega\>*t<around*|(|\<theta\>|)>-n*\<varphi\><around*|(|\<theta\>|)>|]>|}>>>|<row|<cell|>|<cell|>|<cell|<big|oint><frac|\<mathd\>\<theta\><rprime|'>|<wide|\<theta\>|\<dot\>><rprime|'>>*<around*|{|ln*F<rsub|eq>,h|}><around*|(|\<psi\><around*|(|\<theta\><rprime|'>|)>,\<theta\><rprime|'>,v<rsub|<around*|\|||\|>><around*|(|\<theta\><rprime|'>|)>,\<mu\>|)>*exp<around*|{|-i*<around*|[|\<omega\>*t<around*|(|\<theta\><rprime|'>|)>-n*\<varphi\><around*|(|\<theta\><rprime|'>|)>|]>|}>>>|<row|<cell|>|<cell|>|<cell|\<bbb-1\><around*|[|t<around*|(|\<theta\><rprime|'>|)>\<leqslant\>t<around*|(|\<theta\>|)>|]>>>>>
  </eqnarray*>

  Now, the difficult this is to evaluate the two integrals in
  <math|\<theta\>> and <math|\<theta\><rprime|'>> efficiently.

  For this, we consider a RBF basis as

  <\equation*>
    h<rsub|A><around*|(|\<psi\>,\<theta\>|)>=exp<around*|(|-<frac|1|2>*A*<around*|(|\<psi\>-\<psi\><rsub|0>,\<theta\>-\<theta\><rsub|0>|)><rsup|2>|)>
  </equation*>

  with <math|A> a covariance matrix. Therefore, the integral becomes

  <\eqnarray*>
    <tformat|<table|<row|<cell|S<rsub|int><rsup|n*\<omega\>*A>>|<cell|=>|<cell|-<frac|1|2>*<big|int><frac|F<rsub|eq><around*|(|\<mathd\>P<rsub|\<varphi\>>*\<mathd\>E*\<mathd\>\<mu\>|)>|1-exp<around*|{|i*<around*|[|\<omega\>*\<tau\><rsub|b>-n*\<varphi\><rsub|b>|]>|}>>>>|<row|<cell|>|<cell|>|<cell|<big|oint><frac|\<mathd\>\<theta\>|2*\<pi\>>*exp<around*|(|-<frac|1|2>*A<around*|(|\<psi\><around*|(|\<theta\>|)>-\<psi\><rsub|0><rsup|\<dag\>>,\<theta\>-\<theta\><rsub|0><rsup|\<dag\>>|)><rsup|2>|)>*exp<around*|{|i*<around*|[|\<omega\>*t<around*|(|\<theta\>|)>-n*\<varphi\><around*|(|\<theta\>|)>|]>|}>>>|<row|<cell|>|<cell|>|<cell|<big|oint><frac|\<mathd\>\<theta\><rprime|'>|<wide|\<theta\>|\<dot\>><rprime|'>>*exp<around*|(|-<frac|1|2>*A<around*|(|\<psi\><around*|(|\<theta\><rprime|'>|)>-\<psi\><rsub|0>,\<theta\><rprime|'>-\<theta\><rsub|0>|)><rsup|2>|)>*exp<around*|{|-i*<around*|[|\<omega\>*t<around*|(|\<theta\><rprime|'>|)>-n*\<varphi\><around*|(|\<theta\><rprime|'>|)>|]>|}>>>|<row|<cell|>|<cell|>|<cell|\<bbb-1\><around*|[|t<around*|(|\<theta\><rprime|'>|)>\<leqslant\>t<around*|(|\<theta\>|)>|]>>>|<row|<cell|>|<cell|>|<cell|<around*|[|-<around*|{|ln*F<rsub|eq>,\<psi\>|}>*A<rsub|\<psi\>,:><around*|(|\<psi\><around*|(|\<theta\><rprime|'>|)>-\<psi\><rsub|0>,\<theta\><rprime|'>-\<theta\><rsub|0>|)>-<around*|{|ln*F<rsub|eq>,\<theta\>|}>*A<rsub|\<theta\>,:><around*|(|\<psi\><around*|(|\<theta\><rprime|'>|)>-\<psi\><rsub|0>,\<theta\><rprime|'>-\<theta\><rsub|0>|)>+<around*|{|ln*F<rsub|eq>,\<varphi\>|}>*i*n|]>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>>>>
  </eqnarray*>
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|2.1|?>>
    <associate|auto-4|<tuple|2.1|?>>
    <associate|auto-5|<tuple|2.1|?>>
    <associate|auto-6|<tuple|2.1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Computation of trajectories
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|2tab>|1.1<space|2spc>Case of passing particles
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|2tab>|1.2<space|2spc>Case of trapped particles
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|2tab>|1.3<space|2spc>Magnetic configuration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2<space|2spc>Computation of the kernel
      integral <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|2.1<space|2spc>Case for passing particles
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>
    </associate>
  </collection>
</auxiliary>