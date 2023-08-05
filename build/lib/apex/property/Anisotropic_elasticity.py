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

class Anisotropic_elasticity(Property):
    def __init__(self, parameter, inter_param=None):
        parameter["reproduce"] = parameter.get("reproduce", False)
        self.reprod = parameter["reproduce"]
        if not self.reprod:
            if not ("init_from_suffix" in parameter and "output_suffix" in parameter):
            
                self.lattice_type = parameter["lattice_type"]
                self.c11 = parameter["c11"]
                self.c12 = parameter["c12"]
                self.c44 = parameter["c44"]
                self.theta1 = parameter["theta1"]
                self.theta2 = parameter["theta2"]

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
                    if os.path.isfile("eos.json"):
                        os.remove("eos.json")
                    if os.path.islink("eos.json"):
                        os.remove("eos.json")
                    os.symlink(
                        os.path.relpath(os.path.join(init_from_task, "eos.json")),
                        "eos.json",
                    )
                os.chdir(cwd)

        
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

