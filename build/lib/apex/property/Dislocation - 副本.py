import glob
import json
import os
import re

import dpdata
import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.core.surface import generate_all_slabs

import apex.calculator.lib.abacus as abacus
import apex.calculator.lib.vasp as vasp
from apex.property.Property import Property
from apex.property.refine import make_refine
from apex.property.reproduce import make_repro, post_repro
from dflow.python import upload_packages
upload_packages.append(__file__)

class Dislocation(Property):
    def __init__(self, parameter, inter_param=None):
        parameter["reproduce"] = parameter.get("reproduce", False)
        self.reprod = parameter["reproduce"]
        if not self.reprod:
            if not ("init_from_suffix" in parameter and "output_suffix" in parameter):
            
                self.lattice_type = parameter["lattice_type"]
                self.c11 = parameter["c11"]
                self.c12 = parameter["c12"]
                self.c44 = parameter["c44"]
                self.dislocation_type = parameter["dislocation_type"]
                self.theta = parameter["theta"]
                #self.theta2 = parameter["theta2"]

            parameter["cal_type"] = parameter.get("cal_type", "relaxation")
            self.cal_type = parameter["cal_type"]
            default_cal_setting = {
                "relax_pos": True,
                "relax_shape": True,
                "relax_vol": False,
            }
            if "cal_setting" not in parameter:
                parameter["cal_setting"] = default_cal_setting
            else:
                if "relax_pos" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_pos"] = default_cal_setting[
                        "relax_pos"
                    ]
                if "relax_shape" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_shape"] = default_cal_setting[
                        "relax_shape"
                    ]
                if "relax_vol" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_vol"] = default_cal_setting[
                        "relax_vol"
                    ]
            self.cal_setting = parameter["cal_setting"]
        else:
            parameter["cal_type"] = "static"
            self.cal_type = parameter["cal_type"]
            default_cal_setting = {
                "relax_pos": False,
                "relax_shape": False,
                "relax_vol": False,
            }
            if "cal_setting" not in parameter:
                parameter["cal_setting"] = default_cal_setting
            else:
                if "relax_pos" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_pos"] = default_cal_setting[
                        "relax_pos"
                    ]
                if "relax_shape" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_shape"] = default_cal_setting[
                        "relax_shape"
                    ]
                if "relax_vol" not in parameter["cal_setting"]:
                    parameter["cal_setting"]["relax_vol"] = default_cal_setting[
                        "relax_vol"
                    ]
            self.cal_setting = parameter["cal_setting"]
            parameter["init_from_suffix"] = parameter.get("init_from_suffix", "00")
            self.init_from_suffix = parameter["init_from_suffix"]
        self.parameter = parameter
        self.inter_param = inter_param if inter_param != None else {"type": "vasp"}

    def make_confs(self, path_to_work, path_to_equi, refine=False):
        path_to_work = os.path.abspath(path_to_work)
        if os.path.exists(path_to_work):
            #dlog.warning("%s already exists" % path_to_work)
            print("%s already exists" % path_to_work)
        else:
            os.makedirs(path_to_work)
        path_to_equi = os.path.abspath(path_to_equi)

        if "start_confs_path" in self.parameter and os.path.exists(
            self.parameter["start_confs_path"]
        ):
            init_path_list = glob.glob(
                os.path.join(self.parameter["start_confs_path"], "*")
            )
            struct_init_name_list = []
            for ii in init_path_list:
                struct_init_name_list.append(ii.split("/")[-1])
            struct_output_name = path_to_work.split("/")[-2]
            assert struct_output_name in struct_init_name_list
            path_to_equi = os.path.abspath(
                os.path.join(
                    self.parameter["start_confs_path"],
                    struct_output_name,
                    "relaxation",
                    "relax_task",
                )
            )

        cwd = os.getcwd()
        task_list = []
        if self.reprod:
            print("dislocation reproduce starts")
            if "init_data_path" not in self.parameter:
                raise RuntimeError("please provide the initial data path to reproduce")
            init_data_path = os.path.abspath(self.parameter["init_data_path"])
            task_list = make_repro(
                self.inter_param,
                init_data_path,
                self.init_from_suffix,
                path_to_work,
                self.parameter.get("reprod_last_frame", True),
            )
            os.chdir(cwd)

        else:
            if refine:
                print("dislocation refine starts")
                task_list = make_refine(
                    self.parameter["init_from_suffix"],
                    self.parameter["output_suffix"],
                    path_to_work,
                )
                os.chdir(cwd)

                init_from_path = re.sub(
                    self.parameter["output_suffix"][::-1],
                    self.parameter["init_from_suffix"][::-1],
                    path_to_work[::-1],
                    count=1,
                )[::-1]
                task_list_basename = list(map(os.path.basename, task_list))

                for ii in task_list_basename:
                    init_from_task = os.path.join(init_from_path, ii)
                    output_task = os.path.join(path_to_work, ii)
                    os.chdir(output_task)
                    if os.path.isfile("dislocation.json"):
                        os.remove("dislocation.json")
                    if os.path.islink("dislocation.json"):
                        os.remove("dislocation.json")
                    os.symlink(
                        os.path.relpath(os.path.join(init_from_task, "dislocation.json")),
                        "dislocation.json",
                    )
                os.chdir(cwd)

            else:
                if self.inter_param["type"] == "abacus":
                    CONTCAR = abacus.final_stru(path_to_equi)
                    POSCAR = "STRU"
                else:
                    CONTCAR = "CONTCAR"
                    POSCAR = "POSCAR"

                equi_contcar = os.path.join(path_to_equi, CONTCAR)
                if not os.path.exists(equi_contcar):
                    raise RuntimeError("please do relaxation first")

            ###########################################################
            ###########################################################
            ###########################################################
            if self.parameter["dislocation_type"] == "edge":
                if self.theta[0] != 90:
                    print("WARNING: theta should be 90 for edge dislocation, your theta will be set to 90")
                    self.theta = [90]
            if self.parameter["dislocation_type"] == "screw":
                if self.theta[0] != 0:
                    print("WARNING: theta should be 0 for screw dislocation, your theta will be set to 0")
                    self.theta = [0]
            

                dislocation_list = self.theta

                print("dislocation starts")





            ###########################################################
            ###########################################################
            ###########################################################

                os.chdir(path_to_work)
                if os.path.isfile(POSCAR):
                    os.remove(POSCAR)
                if os.path.islink(POSCAR):
                    os.remove(POSCAR)
                os.symlink(os.path.relpath(equi_contcar), POSCAR)
                #           task_poscar = os.path.join(output, 'POSCAR')
                for ii in range(len(dislocation_list)):
                    append_str=str(self.dislocation_type)+"_"+str(self.theta[ii])
                    #output_task = os.path.join(path_to_work, "task.%06d" % ii)
                    output_task = os.path.join(path_to_work, "task.%02d" % ii, append_str)
                    os.makedirs(output_task, exist_ok=True)
                    os.chdir(output_task)
                    for jj in [
                        "INCAR",
                        "POTCAR",
                        "POSCAR",
                        "conf.lmp",
                        "in.lammps",
                        "STRU",
                    ]:
                        if os.path.exists(jj):
                            os.remove(jj)
                    task_list.append(output_task)
                    #dislocation_list[ii].to("POSCAR.tmp", "POSCAR")         #To be modified
                    ################################################################################
                    # generate elastic tensor for atomsk

                    str_line1=str(self.parameter["c11"])+" "+str(self.parameter["c12"])+" "+str(self.parameter["c12"])+" "+"0.0 0.0 0.0 \n"
                    str_line2=str(self.parameter["c12"])+" "+str(self.parameter["c11"])+" "+str(self.parameter["c12"])+" "+"0.0 0.0 0.0 \n"
                    str_line3=str(self.parameter["c12"])+" "+str(self.parameter["c12"])+" "+str(self.parameter["c11"])+" "+"0.0 0.0 0.0 \n"
                    str_line4="0.0 0.0 0.0 "+str(self.parameter["c44"])+" "+"0.0 0.0 \n"
                    str_line5="0.0 0.0 0.0 "+"0.0 "+str(self.parameter["c44"])+" "+"0.0 \n"
                    str_line6="0.0 0.0 0.0 "+"0.0 0.0 "+str(self.parameter["c44"])+"\n"
                 
                    Element = "Cu"
                    a = 3.6




                    ################################################################################
                    # generate POSCAR    
                    if self.parameter["dislocation_type"] == "edge":
                        print("edge dislocation, the angle between dislocation line and Burgers vector is 0")

                        with open("elastic.txt","w") as f:
                            f.write("# Full 6*6 elastic tensor for accessible \n")
                            f.write("elastic\n")

                            f.write(str_line1)
                            f.write(str_line2)
                            f.write(str_line3)
                            f.write(str_line4)
                            f.write(str_line5)
                            f.write(str_line6)       

                        os.system("rm M*")
                        os.system("rm POSCAR")

                        bvector = a/np.sqrt(2)
                        #repx = 500
                        #repy = 200
                        repx = 50
                        repy = 20
                        repz = 1
                        px = bvector*repx/2-bvector/2
                        py = a*repy/2*np.sqrt(3)-a*np.sqrt(3)/6
                        pz = a*repz/2*np.sqrt(2)-a*np.sqrt(2)/4
                        print(px)
                        print(py)
                        print(pz)
                        os.system("atomsk --create fcc %f %s orient [-110] [111] [11-2] -duplicate %d %d %d M_supercell.xsf"%(a,Element,repx,repy,repz))
                        os.system("atomsk M_supercell.xsf -prop elastic.txt -dislocation %f %f edge Z Y %f 0 M_edge.cfg"%(px,py,bvector))
                        os.system("atomsk M_edge.cfg POSCAR")
                    
                    elif self.parameter["dislocation_type"] == "screw":
                        print("screw dislocation, the angle between dislocation line and Burgers vector is 0")

                        with open("elastic.txt","w") as f:
                            f.write("# Full 6*6 elastic tensor for accessible \n")
                            f.write("elastic\n")

                            f.write(str_line1)
                            f.write(str_line2)
                            f.write(str_line3)
                            f.write(str_line4)
                            f.write(str_line5)
                            f.write(str_line6)

                        
                        os.system("rm M*")
                        os.system("rm POSCAR")
                        bvector = a/np.sqrt(2)
                        repx = 100
                        repy = 140
                        repz = 4
                        px = a*repx*np.sqrt(3)/2-a*np.sqrt(3)/6
                        py = a*repy*0.6123724357-a*0.2041241452
                        pz = a*repz*0.3535533905932737622004-a*0.17677669529663688
                        print(px)
                        print(py)
                        print(pz)
                        os.system("atomsk --create fcc %f %s orient [111] [11-2] [-110] -duplicate %d %d %d M_supercell.xsf"%(a,Element,repx,repy,repz))
                        os.system("atomsk M_supercell.xsf -prop elastic.txt -dislocation %f %f screw Z X %f M_screw.cfg"%(px,py,bvector))
                        os.system("atomsk M_screw.cfg POSCAR")

                    
                    elif self.parameter["dislocation_type"] == "mixed":
                        print("mixed dislocation, the angle between dislocation line and Burgers vector is :"+str(self.parameter["theta"]))
                        theta = self.theta[ii]/180*np.pi

                        bvector = a/np.sqrt(2)
                        y = np.array([1, 1, 1])/np.sqrt(3)
                        y_int = np.array([1, 1, 1])

                        b_t = np.array([1, -1, 0])/np.sqrt(2)
                        b_n = np.cross(y, b_t)
                        
                        # z is perpendicular to y, and the angle between z and b is theta
                        z = np.cos(theta) * b_t + np.sin(theta) * b_n
                        z_int = np.round(z*100)

                        x = np.cross(y, z)
                        x_int = np.cross(y_int, z_int)

                        # vector input for atomsk
                        b_atomsk = np.zeros(3)
                        b_atomsk[0] = -bvector*np.sin(theta)
                        b_atomsk[2] = bvector*np.cos(theta)
                        # vector input for insert_dislocation

                        #z = np.cross(y, b)
                        x_direction="["+str(int(x_int[0]))+"_"+str(int(x_int[1]))+"_"+str(int(x_int[2]))+"]"
                        y_direction="["+str(int(y_int[0]))+"_"+str(int(y_int[1]))+"_"+str(int(y_int[2]))+"]"
                        z_direction="["+str(int(z_int[0]))+"_"+str(int(z_int[1]))+"_"+str(int(z_int[2]))+"]"


                        with open("elastic.txt","w") as f:
                            f.write("# Full 6*6 elastic tensor for accessible \n")
                            f.write("elastic\n")

                            f.write(str_line1)
                            f.write(str_line2)
                            f.write(str_line3)
                            f.write(str_line4)
                            f.write(str_line5)
                            f.write(str_line6)

                            f.write("orientations\n")
                            f.write(x_direction + "\n")
                            f.write(y_direction + "\n")
                            f.write(z_direction + "\n")
                        

                        os.system("rm M*")
                        os.system("rm POSCAR")
                        
                        repx = 100
                        repy = 140
                        repz = 4
                        px = a*repx*np.sqrt(3)/2-a*np.sqrt(3)/6
                        py = a*repy*0.6123724357-a*0.2041241452
                        pz = a*repz*0.3535533905932737622004-a*0.17677669529663688
                        print(px)
                        print(py)
                        print(pz)
                        os.system("atomsk --create fcc %f %s orient %s %s %s -duplicate %d %d %d M_supercell.xsf"%(a,Element,x_direction,y_direction,z_direction,repx,repy,repz))
                        os.system("atomsk M_supercell.xsf -prop elastic.txt -dislocation %f %f mixed Z Y %f %f %f M_mixed.cfg"%(px,py,b_atomsk[0],b_atomsk[1],b_atomsk[2]))
                        os.system("atomsk M_mixed.cfg POSCAR")


                    else:
                        raise RuntimeError("Wrong dislocation type, input should be edge, screw or mixed")
                    #############################################################################


                    if self.inter_param["type"] == "abacus":
                        abacus.poscar2stru("POSCAR", self.inter_param, "STRU")
                        os.remove("POSCAR")
                    # record miller
                    dumpfn(self.dislocation_type, "dislocation.json")
                os.chdir(cwd)

        return task_list

    def post_process(self, task_list):
        pass

    def task_type(self):
        return self.parameter["type"]

    def task_param(self):
        return self.parameter

    def _compute_lower(self, output_file, all_tasks, all_res):
        output_file = os.path.abspath(output_file)
        res_data = {}
        ptr_data = os.path.dirname(output_file) + "\n"

        res_data["c11"] = self.parameter["c11"]

        ptr_data += "c11 = " + str(self.parameter["c11"]) + "\n"
        #print("Just for test: ", self.parameter["c11"])

        return res_data, ptr_data

