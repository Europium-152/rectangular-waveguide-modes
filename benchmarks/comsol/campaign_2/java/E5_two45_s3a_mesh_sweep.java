/*
 * E5_two45_s3a_mesh_sweep.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 12 2026, 04:58 by COMSOL 6.2.0.415. */
public class E5_two45_s3a_mesh_sweep {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing\\campaign_2");

    model.label("E5_two45_s3a_mesh_sweep.mph");

    model.param().set("mesh_factor", "10");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 2);

    model.result().table().create("tbl1", "Table");

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("pol1", "Polygon");
    model.component("comp1").geom("geom1").feature("pol1").set("source", "table");
    model.component("comp1").geom("geom1").feature("pol1")
         .set("table", new String[][]{{"0.0000[mm]", "-42.5945[mm]"}, 
         {"0.0000[mm]", "-27.5945[mm]"}, 
         {"22.8600[mm]", "27.5945[mm]"}, 
         {"39.0245[mm]", "43.7589[mm]"}, 
         {"94.2134[mm]", "66.6189[mm]"}, 
         {"109.2134[mm]", "66.6189[mm]"}, 
         {"109.2134[mm]", "43.7589[mm]"}, 
         {"71.3534[mm]", "43.7589[mm]"}, 
         {"22.8600[mm]", "-4.7345[mm]"}, 
         {"22.8600[mm]", "-42.5945[mm]"}});
    model.component("comp1").geom("geom1").create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("r1").set("pos", new String[]{"0", "-42.5945[mm]-0.02"});
    model.component("comp1").geom("geom1").feature("r1").set("size", new String[]{"22.8600[mm]", "2[cm]"});
    model.component("comp1").geom("geom1").create("r2", "Rectangle");
    model.component("comp1").geom("geom1").feature("r2").set("pos", new String[]{"109.2134[mm]", "43.7589[mm]"});
    model.component("comp1").geom("geom1").feature("r2").set("size", new String[]{"2[cm]", "22.8600[mm]"});
    model.component("comp1").geom("geom1").run();
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").material().create("mat1", "Common");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(1, 3);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 1);
    model.component("comp1").physics("emw").feature("port1").selection().set(4);
    model.component("comp1").physics("emw").create("port2", "Port", 1);
    model.component("comp1").physics("emw").feature("port2").selection().set(13);

    model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");

    model.result().table("tbl1").comments("Global Evaluation 1");

    model.component("comp1").view("view1").axis().set("xmin", -0.020222259685397148);
    model.component("comp1").view("view1").axis().set("xmax", 0.14943565428256989);
    model.component("comp1").view("view1").axis().set("ymin", -0.06582483649253845);
    model.component("comp1").view("view1").axis().set("ymax", 0.06984922289848328);

    model.component("comp1").material("mat1").label("Perfect vacuum");
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermeability", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermittivity", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "0[kg/m^3]");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("electricconductivity", new String[]{"0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"});

    model.component("comp1").physics("emw").feature("port1").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port1").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port1").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port2").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port2").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port2").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port2").set("PortType", "Rectangular");

    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", "c_const/56[GHz]/mesh_factor");
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "c_const/56[GHz]/mesh_factor/10");
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("param2", "Parametric");
    model.study("std1").create("freq", "Frequency");

    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").attach("std1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol().create("sol2");
    model.sol("sol2").study("std1");
    model.sol("sol2").label("Parametric Solutions 1");

    model.batch().create("p1", "Parametric");
    model.batch("p1").create("so1", "Solutionseq");
    model.batch("p1").study("std1");

    model.result().numerical().create("gev1", "EvalGlobal");
    model.result().numerical("gev1").set("data", "dset2");
    model.result().create("pg1", "PlotGroup2D");
    model.result().create("pg2", "PlotGroup1D");
    model.result().create("pg3", "SmithGroup");
    model.result("pg1").set("data", "dset2");
    model.result("pg1").create("surf1", "Surface");
    model.result("pg2").set("data", "dset2");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").set("data", "dset2");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");

    model.study("std1").feature("param2").label("Mesh");
    model.study("std1").feature("param2").set("pname", new String[]{"mesh_factor"});
    model.study("std1").feature("param2").set("plistarr", new String[]{"10 20"});
    model.study("std1").feature("param2").set("punit", new String[]{""});
    model.study("std1").feature("freq").set("plist", "30 56");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"30[GHz] 56[GHz]"});
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").set("probesel", "none");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 2");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("aDef").set("cachepattern", true);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"30 56"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("fc1").label("Fully Coupled 1.1");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").runAll();

    model.batch("p1").set("control", "param2");
    model.batch("p1").set("pname", new String[]{"mesh_factor"});
    model.batch("p1").set("plistarr", new String[]{"10 20"});
    model.batch("p1").set("punit", new String[]{""});
    model.batch("p1").set("err", true);
    model.batch("p1").feature("so1").set("seq", "sol1");
    model.batch("p1").feature("so1").set("psol", "sol2");
    model.batch("p1").feature("so1").set("param", new String[]{"\"mesh_factor\",\"10\"", "\"mesh_factor\",\"20\""});
    model.batch("p1").attach("std1");
    model.batch("p1").run();

    model.result().numerical("gev1").set("table", "tbl1");
    model.result().numerical("gev1").set("expr", new String[]{"emw.S11", "emw.S21"});
    model.result().numerical("gev1").set("unit", new String[]{"1", "1"});
    model.result().numerical("gev1").set("descr", new String[]{"S11", "S21"});
    model.result().numerical("gev1").setResult();
    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").feature("surf1").label("Surface");
    model.result("pg1").feature("surf1").set("smooth", "internal");
    model.result("pg1").feature("surf1").set("resolution", "normal");
    model.result("pg2").label("S-parameter (emw)");
    model.result("pg2").set("ylabel", "S-parameter (dB)");
    model.result("pg2").set("ylabelactive", true);
    model.result("pg2").feature("glob1")
         .set("expr", new String[]{"emw.S11dB", "emw.S21dB", "emw.S31dB", "emw.S41dB"});
    model.result("pg2").feature("glob1").set("unit", new String[]{"1", "1", "", ""});
    model.result("pg2").feature("glob1").set("descr", new String[]{"S11", "S21", "S31", "S41"});
    model.result("pg2").feature("glob1").set("titletype", "none");
    model.result("pg2").feature("glob1").set("xdatasolnumtype", "all");
    model.result("pg2").feature("glob1").set("xdata", "expr");
    model.result("pg2").feature("glob1").set("xdataexpr", "mesh_factor");
    model.result("pg2").feature("glob1").set("xdataunit", "");
    model.result("pg2").feature("glob1").set("xdatadescr", "");
    model.result("pg3").label("Smith Plot (emw)");
    model.result("pg3").feature("rgr1").set("titletype", "manual");
    model.result("pg3").feature("rgr1").set("title", "Reflection Graph: S-parameter, Color: Frequency (GHz)");
    model.result("pg3").feature("rgr1").set("linemarker", "point");
    model.result("pg3").feature("rgr1").set("markerpos", "datapoints");
    model.result("pg3").feature("rgr1").feature("col1").set("colortable", "Spectrum");

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
