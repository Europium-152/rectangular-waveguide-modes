/*
 * E1_miter_E_plain.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 12 2026, 02:48 by COMSOL 6.2.0.415. */
public class E1_miter_E_plain {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing\\campaign_2");

    model.label("E1_miter_E_plain.mph");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.result().table().create("tbl1", "Table");

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("wp1", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp1").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("pol1", "Polygon");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pol1").set("source", "table");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pol1")
         .set("table", new String[][]{{"0.0000[mm]", "-20.0800[mm]"}, 
         {"0.0000[mm]", "-5.0800[mm]"}, 
         {"10.1600[mm]", "5.0800[mm]"}, 
         {"25.1600[mm]", "5.0800[mm]"}, 
         {"25.1600[mm]", "-5.0800[mm]"}, 
         {"10.1600[mm]", "-5.0800[mm]"}, 
         {"10.1600[mm]", "-20.0800[mm]"}});
    model.component("comp1").geom("geom1").create("ext1", "Extrude");
    model.component("comp1").geom("geom1").feature("ext1").setIndex("distance", "22.86[mm]", 0);
    model.component("comp1").geom("geom1").feature("ext1").selection("input").set("wp1");
    model.component("comp1").geom("geom1").create("ext2", "Extrude");
    model.component("comp1").geom("geom1").feature("ext2").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext2").setIndex("distance", "0.02", 0);
    model.component("comp1").geom("geom1").feature("ext2").selection("inputface").set("ext1(1)", 2);
    model.component("comp1").geom("geom1").create("ext3", "Extrude");
    model.component("comp1").geom("geom1").feature("ext3").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext3").setIndex("distance", "0.02", 0);
    model.component("comp1").geom("geom1").feature("ext3").selection("inputface").set("ext2(1)", 14);
    model.component("comp1").geom("geom1").run();

    model.component("comp1").material().create("mat1", "Common");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(1, 3);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 2);
    model.component("comp1").physics("emw").feature("port1").selection().set(6);
    model.component("comp1").physics("emw").create("port2", "Port", 2);
    model.component("comp1").physics("emw").feature("port2").selection().set(14);

    model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");

    model.result().table("tbl1").comments("Global Evaluation 1");

    model.component("comp1").view("view1").set("renderwireframe", true);
    model.component("comp1").view("view2").axis().set("xmin", -0.006539292633533478);
    model.component("comp1").view("view2").axis().set("xmax", 0.0316992923617363);
    model.component("comp1").view("view2").axis().set("ymin", -0.021338000893592834);
    model.component("comp1").view("view2").axis().set("ymax", 0.006337999831885099);

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
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", "c_const/36[GHz]/5");
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "c_const/36[GHz]/50");
    model.component("comp1").mesh("mesh1").run();

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
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");

    model.result().numerical().create("gev1", "EvalGlobal");
    model.result().create("pg1", "PlotGroup3D");
    model.result().create("pg2", "PlotGroup1D");
    model.result().create("pg3", "SmithGroup");
    model.result("pg1").create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("expr", "emw.Ex^2+emw.Ey^2");
    model.result("pg1").feature("mslc1").create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");

    model.study("std1").feature("freq").set("plist", "range(8,0.4,12.4) 24 36");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"range(8,0.4,12.4)[GHz] 24[GHz] 36[GHz]"});
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").set("probesel", "none");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 1");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("aDef").set("cachepattern", true);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"range(8,0.4,12.4) 24 36"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("fc1").label("Fully Coupled 1.1");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").set("itrestart", 300);
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").feature("ilDef").label("Incomplete LU 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").label("Multigrid 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("iter", 1);
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").label("Presmoother 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("soDef").label("SOR 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .label("SOR Vector 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").label("Postsmoother 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("soDef").label("SOR 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .label("SOR Vector 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").label("Coarse Solver 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("dDef").label("Direct 2");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1").label("Direct 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1")
         .set("linsolver", "pardiso");
    model.sol("sol1").runAll();

    model.result().numerical("gev1").set("table", "tbl1");
    model.result().numerical("gev1").set("expr", new String[]{"emw.S11", "emw.S21"});
    model.result().numerical("gev1").set("unit", new String[]{"1", "1"});
    model.result().numerical("gev1").set("descr", new String[]{"S11", "S21"});
    model.result().numerical("gev1").setResult();
    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").feature("mslc1").label("Multislice");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("resolution", "normal");
    model.result("pg1").feature("mslc1").feature("filt1").active(false);
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

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
