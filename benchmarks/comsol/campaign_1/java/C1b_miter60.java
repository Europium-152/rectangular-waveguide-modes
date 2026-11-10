/*
 * C1b_60miter.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 10 2026, 07:05 by COMSOL 6.2.0.415. */
public class C1b_60miter {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing");

    model.label("C1b_60miter.mph");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 2);

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").useConstrDim(true);
    model.component("comp1").geom("geom1").create("pol1", "Polygon");
    model.component("comp1").geom("geom1").feature("pol1")
         .set("tableconstr", new String[]{"on", "on", "on", "on", "on", "on", "on", "on"});
    model.component("comp1").geom("geom1").feature("pol1").set("source", "table");
    model.component("comp1").geom("geom1").feature("pol1")
         .set("table", new String[][]{{"0.0000", "-34.7973[mm]"}, 
         {"0.0000", "-19.7973[mm]"}, 
         {"22.8600[mm]", "19.7973[mm]"}, 
         {"35.8504[mm]", "27.2973[mm]"}, 
         {"47.2804[mm]", "7.5000[mm]"}, 
         {"22.8600[mm]", "-6.5991[mm]"}, 
         {"22.8600[mm]", "-34.7973[mm]"}});
    model.component("comp1").geom("geom1").create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("r1").set("pos", new double[]{0.0472804, 0.0075});
    model.component("comp1").geom("geom1").feature("r1").set("rot", 30.00005104308949);
    model.component("comp1").geom("geom1").feature("r1").set("sizeconstr", new String[]{"on", "on"});
    model.component("comp1").geom("geom1").feature("r1").set("size", new String[]{"2[cm]", "22.86[mm]"});
    model.component("comp1").geom("geom1").create("r2", "Rectangle");
    model.component("comp1").geom("geom1").feature("r2").set("pos", new double[]{0, -0.0547973});
    model.component("comp1").geom("geom1").feature("r2").set("sizeconstr", new String[]{"on", "on"});
    model.component("comp1").geom("geom1").feature("r2").set("size", new String[]{"22.86[mm]", "2[cm]"});
    model.component("comp1").geom("geom1").create("coi2", "Coincident");
    model.component("comp1").geom("geom1").feature("coi2").selection("entity1").init(0);
    model.component("comp1").geom("geom1").feature("coi2").selection("entity1").set("r2(1)", 4);
    model.component("comp1").geom("geom1").feature("coi2").selection("entity2").init(0);
    model.component("comp1").geom("geom1").feature("coi2").selection("entity2").set("pol1(1)", 1);
    model.component("comp1").geom("geom1").create("coi1", "Coincident");
    model.component("comp1").geom("geom1").feature("coi1").selection("entity1").init(0);
    model.component("comp1").geom("geom1").feature("coi1").selection("entity1").set("r1(1)", 1);
    model.component("comp1").geom("geom1").feature("coi1").selection("entity2").init(0);
    model.component("comp1").geom("geom1").feature("coi1").selection("entity2").set("pol1(1)", 5);
    model.component("comp1").geom("geom1").run();
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").material().create("mat1", "Common");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(1, 3);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 1);
    model.component("comp1").physics("emw").feature("port1").selection().set(4);
    model.component("comp1").physics("emw").create("port2", "Port", 1);
    model.component("comp1").physics("emw").feature("port2").selection().set(10);

    model.component("comp1").view("view1").axis().set("xmin", -0.044882871210575104);
    model.component("comp1").view("view1").axis().set("xmax", 0.10948377102613449);
    model.component("comp1").view("view1").axis().set("ymin", -0.06461306661367416);
    model.component("comp1").view("view1").axis().set("ymax", 0.047113120555877686);

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

    model.component("comp1").physics("emw").prop("MeshControl").set("SizeControlParameter", "UserDefined");
    model.component("comp1").physics("emw").prop("MeshControl")
         .set("PhysicsControlledMeshMaximumElementSize", "c_const/90[GHz]/5");
    model.component("comp1").physics("emw").feature("port1").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port1").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port1").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port2").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port2").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port2").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port2").set("PortType", "Rectangular");

    model.study().create("std1");
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

    model.result().create("pg1", "PlotGroup2D");
    model.result().create("pg2", "PlotGroup1D");
    model.result().create("pg3", "SmithGroup");
    model.result("pg1").create("surf1", "Surface");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");
    model.result().export().create("data1", "Data");

    model.study("std1").feature("freq").set("plist", "30 60 90");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"30[GHz] 60[GHz] 90[GHz]"});
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").set("probesel", "none");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 2");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("aDef").set("cachepattern", true);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"30 60 90"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("fc1").label("Fully Coupled 1.1");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").runAll();

    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").feature("surf1").label("Surface");
    model.result("pg1").feature("surf1").set("smooth", "internal");
    model.result("pg1").feature("surf1").set("resolution", "normal");
    model.result("pg2").label("S-parameter (emw)");
    model.result("pg2").set("ylabel", "S-parameter (dB)");
    model.result("pg2").set("ylabelactive", true);
    model.result("pg2").feature("glob1").set("expr", new String[]{"emw.S11dB", "emw.S21dB"});
    model.result("pg2").feature("glob1").set("unit", new String[]{"1", "1"});
    model.result("pg2").feature("glob1").set("descr", new String[]{"S11", "S21"});
    model.result("pg2").feature("glob1").set("titletype", "none");
    model.result("pg2").feature("glob1").set("xdata", "expr");
    model.result("pg2").feature("glob1").set("xdataexpr", "freq");
    model.result("pg2").feature("glob1").set("xdataunit", "GHz");
    model.result("pg2").feature("glob1").set("xdatadescr", "Frequency");
    model.result("pg3").label("Smith Plot (emw)");
    model.result("pg3").feature("rgr1").set("titletype", "manual");
    model.result("pg3").feature("rgr1").set("title", "Reflection Graph: S-parameter, Color: Frequency (GHz)");
    model.result("pg3").feature("rgr1").set("linemarker", "point");
    model.result("pg3").feature("rgr1").set("markerpos", "datapoints");
    model.result("pg3").feature("rgr1").feature("col1").set("colortable", "Spectrum");
    model.result().export("data1").set("expr", new String[]{"emw.S11", "emw.S21"});
    model.result().export("data1").set("unit", new String[]{"1", "1"});
    model.result().export("data1").set("descr", new String[]{"S11", "S21"});
    model.result().export("data1")
         .set("filename", "C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing\\results_files\\C1b_60miter.txt");
    model.result().export("data1").set("location", "grid");
    model.result().export("data1").set("gridx2", 0);
    model.result().export("data1").set("gridy2", -0.03);

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
