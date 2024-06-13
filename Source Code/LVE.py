import numpy as np
from typedefs import *
from calc import *
import matplotlib.pyplot as plt
import copy
from typedef2 import *
from solve import lin_solve
from tkinter import * 
import time

def LVE(surf, curfield, app, nsteps=500, dtstar= 1,animStep=30,save_file_force_data=False,file_name_file_force_data='',save_flow_force_data=False,tev_file_name=''):
    # Initialize variables
    import numpy as np
    mat = [[0 for _ in range(1)] for _ in range(8)] # loads output matrix
    prevCirc = np.zeros(surf.ndiv-1)
    circChange = np.zeros(surf.ndiv-1)
    vor_loc = np.zeros((surf.ndiv-1, 2)) # (x1, z1; x2, z2...) Inertial frame
    coll_loc = np.zeros((surf.ndiv-1, 2)) # (x1, z1; x2, z2...)
    global RHS
    RHS = np.zeros((surf.ndiv))
    # relVel = np.zeros((surf.ndiv-1, 2))
    global relVel
    relVel=[[0 for _ in range(2)] for _ in range(surf.ndiv-1)]
    test=np.empty((1, 2))
    IFR_vor = np.zeros_like(vor_loc) # Global frame
    IFR_field =TwoDFlowField(ConstDef(0),ConstDef(0))
    IFR_surf = copy.deepcopy(surf)
    global dp
    dp = np.zeros((surf.ndiv-1, 1)) # change in panel pressure
    tev_loc = [] #[[0 for _ in range(1)] for _ in range(4)]
    #tev_loc = ["# t*","alpha","lesp","cl","cd","cm"]
    lev_loc = []# [[0 for _ in range(1)] for _ in range(4)]
    #lev_loc = ["# t*","alpha","lesp","cl","cd","cm"]

    if save_file_force_data == True:
        with open(f"{file_name_file_force_data}.dat", "w") as file:
            file.write(f'# t*\t\talpha\t\tLESP\t\tCl\t\tCd\t\tCm\n')

    if save_flow_force_data == True:
        # Clear content of alphamotion.txt
        with open(f"{tev_file_name}_bv.dat", "w") as file:
            pass  # No need to write anything, just opening in write mode clears the file

        # Clear content of tev_loc.txt
        with open(f"{tev_file_name}_tev.dat", "w") as file:
            pass  # No need to write anything, just opening in write mode clears the file

        # Clear content of lev_loc.txt
        with open(f"{tev_file_name}_lev.dat", "w") as file:
            pass  # No need to write anything, just opening in write mode clears the file

    # panel geometry
    ds, cam_slope, n, tau, vor_loc, coll_loc = panelGeo(surf)
    ## Critical gamma (LEV)
    x = 1 - 2*ds[0] / surf.c
    surf_lespcrit = np.array(surf.lespcrit)
    surf_kinem_u = np.array(surf.kinem.u)
    surf_c = np.array(surf.c)
    x = np.array(x)
    gamma_crit = (surf_lespcrit / 1.13) * ( surf_kinem_u * surf_c * (math.acos(x) + np.sin(math.acos(x))))
    gamma_crit = gamma_crit[0]

    ## Refresh vortex values
    surf = refresh_vortex(surf, vor_loc)

    ## Time Stepping Loop
    t = -dtstar
    tstart = time.time()
    for i in range(nsteps+1):
        if i%50 == 0:
            print(nsteps, i)
    
        t = round(t+dtstar , 4)
        # Update Kinematics
        surf = update_kinem(surf, t, dtstar)

        # Inertial reference frame
        IFR_vor[:, 0], IFR_vor[:, 1] = IFR(surf, vor_loc[:, 0], vor_loc[:, 1], t)
        IFR_surf = refresh_vortex(IFR_surf, IFR_vor)
        
        ## TEV setup
        IFR_field.tev = place_tev2(IFR_surf, IFR_field, dtstar, t)

        x_w, z_w = BFR(surf, IFR_field.tev[0][-1], IFR_field.tev[1][-1], t)
        
        global a
        a = influence_coeff(surf, coll_loc, n, x_w, z_w)
        ## RHS Vector
        RHS[-1] = np.sum(prevCirc) # previous total circulation
        
        # u_w, w_w
        if (curfield.tev[0] > [0]): # Trailing edge vortexs exist
            surf.uind[:-1], surf.wind[:-1] =ind_velw(np.array([row1 + row2 for row1, row2 in zip(curfield.tev, curfield.lev)]), np.array(coll_loc[:, 0]), np.array(coll_loc[:, 1]))

        for j in range(surf.ndiv-1):
            alpha = surf.kinem.alpha
            A_v=([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
            x_v=([[surf.kinem.u],[ -surf.kinem.hdot]])
            B_v=([[-surf.kinem.alphadot*coll_loc[j][1]], [surf.kinem.alphadot * (coll_loc[j][0] - surf.pvt)]])
            relVel[j][:] = np.dot(A_v,x_v)+B_v
            relVel=[[element.tolist() if isinstance(element, np.ndarray) else element for element in sublist] for sublist in relVel]
            RHS[j] = -((relVel[j][0] + surf.uind[j]) * n[j, 0] + (relVel[j][1] + surf.wind[j]) * n[j, 1])

        # Vortex Strengths a*circ = RHS
        global circ
        circ = lin_solve(a, RHS)
        
        # Circulation changes before each panel
        for j in range(1,surf.ndiv):
            gamma1 = sum(prevCirc[:j-1])
            gamma2 = sum(circ[:j-1])
            circChange[j-1] = (gamma2 - gamma1) / dtstar
            
        # Test for LEV criteria
        #print(circ[0])

        if abs(circ[0]) > gamma_crit:

            gamma_crit_use = abs(circ[0])/circ[0] * gamma_crit #~gamma
            
            # Place LEV
            IFR_field.lev= place_lev2(surf, IFR_field, dtstar, t,i)

            if surf.levflag[0]==0:
                print("LEV start batch:")
            
            surf.levflag[0] = 1  # Set LEV flag to track constant LEV streams

            # Convert coords to BFR
            x_lev, z_lev = BFR(surf, IFR_field.lev[0][-1], IFR_field.lev[1][-1], t) #~x_lev, z_lev
            global a1
            a, a1 = mod_influence_coeff(surf, coll_loc, n, x_w, z_w, x_lev, z_lev)
            # Recalculate RHS
            # u_w, w_w
            if len(curfield.tev[0]) > 1:  # Trailing edge vortexs exist
                surf.uind[0:-1], surf.wind[0:-1] = ind_velw(np.array([row1 + row2 for row1, row2 in zip(curfield.tev, curfield.lev)]), np.array(coll_loc[:, 0]), np.array(coll_loc[:, 1]))

            for j in range(surf.ndiv - 1):
                # RHS = -[U_t + u_w, W_t + w_w] dot n
                RHS[j] = -((relVel[j][0] + surf.uind[j]) * n[j, 0] + (relVel[j][1] + surf.wind[j]) * n[j, 1]) - gamma_crit_use * a1[j]
                
            RHS[-1] -= gamma_crit_use  # first RHS - gamma_crit_use*a1[end] (which is always 1)

            circ = lin_solve(a, RHS)  # Solve new circulations

            IFR_field.lev[2][-1] = circ[-1]  # store LEV strength
            circ[1:] = circ[0:-1]  # shift circs
            circ[0] = gamma_crit_use
        else:
            surf.levflag[0] = 0
            
        prevCirc = circ[:-1]

        for j in range(0,surf.ndiv - 1):  # Set bv circs
            surf.bv[2][j] = circ[j]
            IFR_surf.bv[2][j] = circ[j]


        if len(IFR_field.tev[0]) > 1:  # Set TEV circ
            IFR_field.tev[2][-1] = circ[-1]

            
        if i > 0:  # Force Calculation
            ## Pressures (Change in pressure for each panel)
            for j in range(len(dp)):
                dp[j] = surf.rho * (((relVel[j][0] + surf.uind[j]) * tau[j, 0] + (relVel[j][1] + surf.wind[j]) * tau[j, 1]) * circ[j] / ds[j] + circChange[j])
            ## Loads
            # Cn normal
            ndem = 0.5 * surf.rho * surf.kinem.u ** 2 * surf.c  # non-dimensionalization constant
            cn = sum(dp[j] * ds[j] for j in range(len(dp))) / ndem
            cn=cn[0]
            # LESP
            x = 1 - 2 * ds[0] / surf.c
            surf.a0[0] = 1.13 * circ[0] / (surf.kinem.u * surf.c * (math.acos(x) + math.sin(math.acos(x))))
            # Cs suction
            cs = 2 * math.pi * surf.a0[0] ** 2
            # Cl lift
            alpha = surf.kinem.alpha
            cl = cn * math.cos(alpha) + cs * math.sin(alpha)
            # Cd drag
            cd = cn * math.sin(alpha) - cs * math.cos(alpha)
            # Cm moment
            ndem = ndem * surf.c  # change ndem to moment ndem
            # Cm = Cmz + Cmx
            cm = -sum(dp[j] * ds[j] * math.cos(math.radians(cam_slope[j])) * (vor_loc[j, 0] - surf.pvt) for j in range(len(dp))) / ndem + sum(dp[j] * ds[j] * math.sin(math.radians(cam_slope[j])) * vor_loc[j, 1] for j in range(len(dp))) / ndem
            cm=cm[0]

            test=[t, surf.kinem.alpha, surf.a0[0], cl, cd, cm]

            if (len(mat) < 6):
                mat = test
            else:
                mat = [row + [new_value] for row, new_value in zip(mat, test)]  # Pressures and loads
            
            if save_file_force_data == True:
                with open(f"{file_name_file_force_data}.dat", "a") as file:
                    # Convert the entire test list to string and write it to the file
                    file.write("\t".join(map(str, test)) + "\n")
                    # Write an empty line after each row
        
        IFR_field = wakeroll(IFR_surf, IFR_field, dtstar)
        # Convert IFR_field values to curfield (IFR -> BFR)
        if(curfield.tev[0] == [0]):
            curfield.tev[2][0]=IFR_field.tev[2][-1]
            curfield.tev[3][0]=0.02 * surf.c
        else:
            curfield.tev= [row + [new_value] for row, new_value in zip(curfield.tev, [0, 0, IFR_field.tev[2][-1], 0.02 * surf.c, 0., 0.])]

        if (IFR_field.lev[0]) != [0] and surf.levflag[0] == 1:
            if(curfield.lev[0] == [0]):
                curfield.lev[2][0]=IFR_field.lev[2][-1]
                curfield.lev[3][0]=0.02 * surf.c
            else:
                curfield.lev=[row + [new_value] for row, new_value in zip(curfield.lev, [0, 0, IFR_field.lev[2][-1], 0.02 * surf.c, 0., 0.])]
                
        curfield = vor_BFR(surf, round(t + dtstar,4), IFR_field, curfield)  # Calculate IFR positions for next iteration
    
        if (i % int(animStep) == 0 and i >=0) or (i==0) or (i == nsteps): 

            alphamotion=[]
            camX, camZ = IFR(surf,surf.x,surf.cam,t)
            app.update_graphs(mat, camX, camZ, IFR_field.tev, IFR_field.lev, surf.kinem.alpha, t)
            app.root.update_idletasks()
            
            # Append the IFR values in separate loops
            for j in range(len(camX)-1):
                alphamotion.append([ (t),  (camX[j]),  (camZ[j]), (surf.bv[2][j])])

            for j in range(len(IFR_field.tev[0])):
                tev_loc.append([ (t),  (IFR_field.tev[0][j]),  (IFR_field.tev[1][j]),  (IFR_field.tev[2][j])])

            for j in range(len(IFR_field.lev[0])):
                lev_loc.append([ (t),  (IFR_field.lev[0][j]),  (IFR_field.lev[1][j]),  (IFR_field.lev[2][j])])

            
            if save_flow_force_data == True:
                # Open a text file for writing alphamotion
                with open(f"{tev_file_name}_bv.dat", "a") as file:
                    # Write header
                    file.write(f"# iter={i}     t*={t}\n")
                    
                    # Iterate through alphamotion list
                    for row in alphamotion:
                        # Convert each row to string and write to the file
                        file.write("\t ".join(map(str, row)) + "\n")

                # Open a text file for writing tev_loc
                with open(f"{tev_file_name}_tev.dat", "a") as file:
                    # Write header
                    file.write(f"# iter={i}     t*={t}\n")
                    
                    # Iterate through tev_loc list
                    for row in tev_loc:
                        # Convert each row to string and write to the file
                        file.write(" \t".join(map(str, row)) + "\n")

                # Open a text file for writing lev_loc
                with open(f"{tev_file_name}_lev.dat", "a") as file:
                    # Write header
                    file.write(f"# iter={i}     t*={t}\n")
                    
                    # Iterate through lev_loc list
                    for row in lev_loc:
                        # Convert each row to string and write to the file
                        file.write("\t ".join(map(str, row)) + "\n")

            tev_loc=[]
            lev_loc=[]    
            
    tstop= time.time()
    return tstop-tstart