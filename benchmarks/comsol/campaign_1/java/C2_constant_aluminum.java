/*
 * C2_constant_aluminum.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 10 2026, 07:12 by COMSOL 6.2.0.415. */
public class C2_constant_aluminum {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing");

    model.label("C2_constant_aluminum.mph");

    model.param().set("wr90_long", "0.9[in]", "WR90 broad wall");
    model.param().set("wr90_short", "0.4[in]", "WR90 narrow wall");
    model.param().set("length", "15[cm]", "Bend length");
    model.param().set("angle", "60[deg]");
    model.param().set("curvature_radius", "length / angle");
    model.param().set("pml_thickness", "2[cm]", "PML thickness");
    model.param().set("side_a", "wr90_short");
    model.param().set("side_b", "wr90_long");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("wp1", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp1").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ca1", "CircularArc");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ca1")
         .set("r", "curvature_radius - side_a/2");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ca1").set("angle2", "angle");
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ca2", "CircularArc");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ca2")
         .set("r", "curvature_radius + side_a/2");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ca2").set("angle2", "angle");
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ls1", "LineSegment");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls1").selection("vertex1")
         .set("ca1(1)", 2);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls1").selection("vertex2")
         .set("ca2(1)", 2);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ls2", "LineSegment");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls2").selection("vertex1")
         .set("ca1(1)", 1);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls2").selection("vertex2")
         .set("ca2(1)", 1);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("csol1", "ConvertToSolid");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("csol1").selection("input")
         .set("ca1", "ca2", "ls1", "ls2");
    model.component("comp1").geom("geom1").create("ext1", "Extrude");
    model.component("comp1").geom("geom1").feature("ext1").setIndex("distance", "side_b", 0);
    model.component("comp1").geom("geom1").feature("ext1").selection("input").set("wp1");
    model.component("comp1").geom("geom1").create("ext2", "Extrude");
    model.component("comp1").geom("geom1").feature("ext2").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext2").setIndex("distance", "pml_thickness", 0);
    model.component("comp1").geom("geom1").feature("ext2").selection("inputface").set("ext1(1)", 6);
    model.component("comp1").geom("geom1").create("ext3", "Extrude");
    model.component("comp1").geom("geom1").feature("ext3").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext3").setIndex("distance", "pml_thickness", 0);
    model.component("comp1").geom("geom1").feature("ext3").selection("inputface").set("ext2(1)", 1);
    model.component("comp1").geom("geom1").run();
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").material().create("mat2", "Common");
    model.component("comp1").material().create("mat3", "Common");
    model.component("comp1").material("mat3").selection().geom("geom1", 2);
    model.component("comp1").material("mat3").selection().set(7, 8, 9, 10);
    model.component("comp1").material("mat3").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material("mat3").propertyGroup().create("Murnaghan", "Murnaghan");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem().create("pml2", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(3);
    model.component("comp1").coordSystem("pml2").selection().set(1);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 2);
    model.component("comp1").physics("emw").feature("port1").selection().set(15);
    model.component("comp1").physics("emw").create("port2", "Port", 2);
    model.component("comp1").physics("emw").feature("port2").selection().set(6);
    model.component("comp1").physics("emw").create("imp1", "Impedance", 2);
    model.component("comp1").physics("emw").feature("imp1").selection().set(7, 8, 9, 10);

    model.component("comp1").view("view1").set("renderwireframe", true);
    model.component("comp1").view("view2").axis().set("xmin", -0.12661419808864594);
    model.component("comp1").view("view2").axis().set("xmax", 0.5809667706489563);
    model.component("comp1").view("view2").axis().set("ymin", -0.023278504610061646);
    model.component("comp1").view("view2").axis().set("ymax", 0.488848477602005);

    model.component("comp1").material("mat2").label("Perfect vacuum");
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "");
    model.component("comp1").material("mat2").propertyGroup("def").set("relpermeability", "");
    model.component("comp1").material("mat2").propertyGroup("def").set("relpermittivity", "");
    model.component("comp1").material("mat2").propertyGroup("def").set("electricconductivity", "");
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "0[kg/m^3]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("electricconductivity", new String[]{"0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"});
    model.component("comp1").material("mat3").label("Aluminum");
    model.component("comp1").material("mat3").set("family", "aluminum");
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat3").propertyGroup("def").set("heatcapacity", "900[J/(kg*K)]");
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("thermalconductivity", new String[]{"238[W/(m*K)]", "0", "0", "0", "238[W/(m*K)]", "0", "0", "0", "238[W/(m*K)]"});
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("electricconductivity", new String[]{"3.774e7[S/m]", "0", "0", "0", "3.774e7[S/m]", "0", "0", "0", "3.774e7[S/m]"});
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"23e-6[1/K]", "0", "0", "0", "23e-6[1/K]", "0", "0", "0", "23e-6[1/K]"});
    model.component("comp1").material("mat3").propertyGroup("def").set("density", "2700[kg/m^3]");
    model.component("comp1").material("mat3").propertyGroup("Enu").set("E", "70[GPa]");
    model.component("comp1").material("mat3").propertyGroup("Enu").set("nu", "0.33");
    model.component("comp1").material("mat3").propertyGroup("Murnaghan").set("l", "-250[GPa]");
    model.component("comp1").material("mat3").propertyGroup("Murnaghan").set("m", "-330[GPa]");
    model.component("comp1").material("mat3").propertyGroup("Murnaghan").set("n", "-350[GPa]");

    model.component("comp1").physics("emw").prop("MeshControl").set("SizeControlParameter", "UserDefined");
    model.component("comp1").physics("emw").prop("MeshControl")
         .set("PhysicsControlledMeshMaximumElementSize", "c_const/60[GHz]/5");
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
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");

    model.result().create("pg1", "PlotGroup3D");
    model.result().create("pg2", "PlotGroup1D");
    model.result().create("pg3", "SmithGroup");
    model.result().create("pg4", "PlotGroup3D");
    model.result("pg1").create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");
    model.result("pg4").create("mslc1", "Multislice");
    model.result("pg4").feature("mslc1").set("expr", "emw.Ez");
    model.result("pg4").feature("mslc1").create("filt1", "Filter");
    model.result("pg4").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result().export().create("data1", "Data");

    model.study("std1").feature("freq").set("plist", "range(18,2,60)");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"range(18,2,60)[GHz]"});
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").set("probesel", "none");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 1");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("aDef").set("cachepattern", true);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"range(18,2,60)"});
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

    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("looplevel", new int[]{17});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").feature("mslc1").label("Multislice");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("resolution", "normal");
    model.result("pg2").label("S-parameter (emw)");
    model.result("pg2").set("xlabel", "Frequency (GHz)");
    model.result("pg2").set("ylabel", "S-parameter (dB)");
    model.result("pg2").set("ylabelactive", true);
    model.result("pg2").set("xlabelactive", false);
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
    model.result("pg4").label("Electric Field (emw) 1");
    model.result("pg4").set("looplevel", new int[]{17});
    model.result("pg4").set("frametype", "spatial");
    model.result("pg4").set("showlegendsmaxmin", true);
    model.result("pg4").feature("mslc1").label("Multislice");
    model.result("pg4").feature("mslc1").set("xnumber", "0");
    model.result("pg4").feature("mslc1").set("ynumber", "0");
    model.result("pg4").feature("mslc1").set("smooth", "internal");
    model.result("pg4").feature("mslc1").set("resolution", "normal");
    model.result().export("data1").set("expr", new String[]{"emw.S11", "emw.S21"});
    model.result().export("data1").set("unit", new String[]{"1", "1"});
    model.result().export("data1").set("descr", new String[]{"S11", "S21"});
    model.result().export("data1")
         .set("filename", "C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing\\results_files\\C2_constant_aluminum.txt");
    model.result().export("data1").set("location", "grid");
    model.result().export("data1").set("gridx3", 0);
    model.result().export("data1").set("gridy3", 0);
    model.result().export("data1").set("gridz3", 0);

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
