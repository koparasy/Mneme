
double dFphi_ref(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}


double GradientX_ref(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}


double GradientY_ref(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}


double GradientZ_ref(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}


double Divergence_ref(double phix[][DATAYSIZE][DATAXSIZE], 
                  double phiy[][DATAYSIZE][DATAXSIZE],
                  double phiz[][DATAYSIZE][DATAXSIZE], 
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX_ref(phix,dx,dy,dz,x,y,z) + 
         GradientY_ref(phiy,dx,dy,dz,x,y,z) +
         GradientZ_ref(phiz,dx,dy,dz,x,y,z);
}


double Laplacian_ref(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}


double An_ref(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}


double Wn_ref(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An_ref(phix,phiy,phiz,epsilon));
}


double taun_ref(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An_ref(phix,phiy,phiz,epsilon));
}


double dFunc_ref(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

void calculateForce_ref(double phi[][DATAYSIZE][DATAXSIZE], 
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{
  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {

        if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
            (iz < (DATAZSIZE-1)) && (ix > (0)) && 
            (iy > (0)) && (iz > (0))) {

          double phix = GradientX_ref(phi,dx,dy,dz,ix,iy,iz);
          double phiy = GradientY_ref(phi,dx,dy,dz,ix,iy,iz);
          double phiz = GradientZ_ref(phi,dx,dy,dz,ix,iy,iz);
          double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
          double c = 16.0 * W0 * epsilon;
          double w = Wn_ref(phix,phiy,phiz,epsilon,W0);
          double w2 = SQ(w);
          

          Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc_ref(phix,phiy,phiz);
          Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc_ref(phiy,phiz,phix);
          Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc_ref(phiz,phix,phiy);
        }
        else
        {
          Fx[ix][iy][iz] = 0.0;
          Fy[ix][iy][iz] = 0.0;
          Fz[ix][iy][iz] = 0.0;
        }
      }
    }
  }
}

// device function to set the 3D volume
void allenCahn_ref(double phinew[][DATAYSIZE][DATAXSIZE], 
               double phiold[][DATAYSIZE][DATAXSIZE],
               double uold[][DATAYSIZE][DATAXSIZE],
               double Fx[][DATAYSIZE][DATAXSIZE],
               double Fy[][DATAYSIZE][DATAXSIZE],
               double Fz[][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz)
{
  #pragma omp parallel for collapse(3)
  for (int ix = 1; ix < DATAXSIZE-1; ix++) {
    for (int iy = 1; iy < DATAYSIZE-1; iy++) {
      for (int iz = 1; iz < DATAZSIZE-1; iz++) {

        double phix = GradientX_ref(phiold,dx,dy,dz,ix,iy,iz);
        double phiy = GradientY_ref(phiold,dx,dy,dz,ix,iy,iz);
        double phiz = GradientZ_ref(phiold,dx,dy,dz,ix,iy,iz); 

        phinew[ix][iy][iz] = phiold[ix][iy][iz] + 
         (dt / taun_ref(phix,phiy,phiz,epsilon,tau0)) * 
         (Divergence_ref(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - 
          dFphi_ref(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
      }
    }
  }
}

void boundaryConditionsPhi_ref(double phinew[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {

        if (ix == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (ix == DATAXSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == DATAYSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iz == DATAZSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
      }
    }
  }
}

void thermalEquation_ref(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz)
{
  #pragma omp parallel for collapse(3)
  for (int ix = 1; ix < DATAXSIZE-1; ix++) {
    for (int iy = 1; iy < DATAYSIZE-1; iy++) {
      for (int iz = 1; iz < DATAZSIZE-1; iz++) {

        unew[ix][iy][iz] = uold[ix][iy][iz] + 
          0.5*(phinew[ix][iy][iz]-
               phiold[ix][iy][iz]) +
          dt * D * Laplacian_ref(uold,dx,dy,dz,ix,iy,iz);
      }
    }
  }
}

void boundaryConditionsU_ref(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {

        if (ix == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (ix == DATAXSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iy == DATAYSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == 0){
          unew[ix][iy][iz] =  -delta;
        }
        else if (iz == DATAZSIZE-1){
          unew[ix][iy][iz] =  -delta;
        }
      }
    }
  }
}

void swapGrid_ref(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < DATAXSIZE; ix++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int iz = 0; iz < DATAZSIZE; iz++) {
        double tmp = cnew[ix][iy][iz];
        cnew[ix][iy][iz] = cold[ix][iy][iz];
        cold[ix][iy][iz] = tmp;
      }
    }
  }
}

void reference(nRarray *phi_ref, nRarray *u_ref, int vol, int num_steps)
{
  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  nRarray *d_phiold;  // storage for result computed on device
  nRarray *d_phinew;
  nRarray *d_uold;
  nRarray *d_unew;
  nRarray *d_Fx;
  nRarray *d_Fy;
  nRarray *d_Fz;

  // allocate buffers
  d_phiold = (nRarray*) malloc (vol*sizeof(double));
  d_phinew = (nRarray*) malloc (vol*sizeof(double));
  d_uold = (nRarray*) malloc (vol*sizeof(double));
  d_unew = (nRarray*) malloc (vol*sizeof(double));
  d_Fx = (nRarray*) malloc (vol*sizeof(double));
  d_Fy = (nRarray*) malloc (vol*sizeof(double));
  d_Fz = (nRarray*) malloc (vol*sizeof(double));
  
  memcpy(d_phiold, phi_ref, (vol*sizeof(double)));
  memcpy(d_uold, u_ref, (vol*sizeof(double)));

  int t = 0;

  while (t <= num_steps) {

    calculateForce_ref(d_phiold,d_Fx,d_Fy,d_Fz,
                   dx,dy,dz,epsilon,W0,tau0);

    allenCahn_ref(d_phinew,d_phiold,d_uold,
              d_Fx,d_Fy,d_Fz,
              epsilon,W0,tau0,lambda,
              dt,dx,dy,dz);

    boundaryConditionsPhi_ref(d_phinew);

    thermalEquation_ref(d_unew,d_uold,d_phinew,d_phiold,
                    D,dt,dx,dy,dz);

    boundaryConditionsU_ref(d_unew,delta);

    swapGrid_ref(d_phinew, d_phiold);

    swapGrid_ref(d_unew, d_uold);

    t++;
  }

  memcpy(phi_ref, d_phiold, (vol*sizeof(double)));
  memcpy(u_ref, d_uold, (vol*sizeof(double)));

  free(d_phiold);
  free(d_phinew);
  free(d_uold);
  free(d_unew);
  free(d_Fx);
  free(d_Fy);
  free(d_Fz);
}
