/*
 * hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 5 2026, 02:01 by COMSOL 6.2.0.415. */
public class hyperbolic_secant_E_bend_60_deg_lowfreq_parametric {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\hyperbolic_secant\\60_degree");

    model.label("hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.mph");

    model.param().set("wr90_long", "0.9[in]", "WR90 broad wall");
    model.param().set("wr90_short", "0.4[in]", "WR90 narrow wall");
    model.param().set("pml_thickness", "2[cm]", "PML thickness");
    model.param().set("wr90_lead", "2[cm]", "WR90 straight section lead");
    model.param().set("bend_angle", "58[deg]", "Total deflection angle");
    model.param().set("r0", "0.055[m]");
    model.param().set("theta", "60[deg]");
    model.param().set("S", "4");
    model.param().set("beta", "4/theta/r0*atan(tanh(S/2))");
    model.param().set("side_a", "wr90_short");
    model.param().set("side_b", "wr90_long");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.result().table().create("evl3", "Table");

    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh().create("mesh2");
    model.component("comp1").mesh().create("mesh3");
    model.component("comp1").mesh().create("mesh4");

    model.component("comp1").geom("geom1").create("wp1", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp1").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("pc1", "ParametricCurve");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1").set("parname", "z");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1").set("parmin", "-S/beta");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1").set("parmax", "S/beta");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1")
         .set("coord", new String[]{"integrate(cos(2/r0/beta*(atan(tanh(u*beta/2))+atan(tanh(S/2)))), u, -S/beta, z)+side_a/2*cos(2/r0/beta*(atan(tanh(z*beta/2))+atan(tanh(S/2)))+pi/2)", "integrate(sin(2/r0/beta*(atan(tanh(u*beta/2))+atan(tanh(S/2)))), u, -S/beta, z)+side_a/2*sin(2/r0/beta*(atan(tanh(z*beta/2))+atan(tanh(S/2)))+pi/2)"});
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1").set("rtol", 1.0E-4);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc1").set("maxknots", 100);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("pc2", "ParametricCurve");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2").set("parname", "z");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2").set("parmin", "-S/beta");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2").set("parmax", "S/beta");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2")
         .set("coord", new String[]{"integrate(cos(2/r0/beta*(atan(tanh(u*beta/2))+atan(tanh(S/2)))), u, -S/beta, z)-side_a/2*cos(2/r0/beta*(atan(tanh(z*beta/2))+atan(tanh(S/2)))+pi/2)", "integrate(sin(2/r0/beta*(atan(tanh(u*beta/2))+atan(tanh(S/2)))), u, -S/beta, z)-side_a/2*sin(2/r0/beta*(atan(tanh(z*beta/2))+atan(tanh(S/2)))+pi/2)"});
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2").set("rtol", 1.0E-4);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("pc2").set("maxknots", 100);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ls1", "LineSegment");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls1").selection("vertex1")
         .set("pc1(1)", 1);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls1").selection("vertex2")
         .set("pc2(1)", 1);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("ls2", "LineSegment");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls2").selection("vertex1")
         .set("pc1(1)", 2);
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ls2").selection("vertex2")
         .set("pc2(1)", 2);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("csol1", "ConvertToSolid");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("csol1").selection("input")
         .set("ls1", "ls2", "pc1", "pc2");
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

    model.component("comp1").view("view1").clip().create("plane1", "ClipPlane");
    model.view().create("view3", 2);

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("eta", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("Cp", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("rho", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("k", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("cs", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("an1", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("an2", "Analytic");
    model.component("comp1").material("mat1").propertyGroup().create("RefractiveIndex", "Refractive index");
    model.component("comp1").material("mat1").propertyGroup().create("NonlinearModel", "Nonlinear model");
    model.component("comp1").material("mat1").propertyGroup().create("idealGas", "Ideal gas");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func().create("Cp", "Piecewise");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem().create("pml2", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(3);
    model.component("comp1").coordSystem("pml2").selection().set(1);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 2);
    model.component("comp1").physics("emw").feature("port1").selection().set(11);
    model.component("comp1").physics("emw").create("port2", "Port", 2);
    model.component("comp1").physics("emw").feature("port2").selection().set(6);

    model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
    model.component("comp1").mesh("mesh1").feature("ftet1").selection().geom("geom1");
    model.component("comp1").mesh("mesh1").feature("ftet1").create("size1", "Size");
    model.component("comp1").mesh("mesh2").create("ftet1", "FreeTet");
    model.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1");
    model.component("comp1").mesh("mesh2").feature("ftet1").create("size1", "Size");
    model.component("comp1").mesh("mesh3").create("ftet1", "FreeTet");
    model.component("comp1").mesh("mesh3").feature("ftet1").selection().geom("geom1");
    model.component("comp1").mesh("mesh3").feature("ftet1").create("size1", "Size");
    model.component("comp1").mesh("mesh4").create("ftet1", "FreeTet");
    model.component("comp1").mesh("mesh4").feature("ftet1").selection().geom("geom1");
    model.component("comp1").mesh("mesh4").feature("ftet1").create("size1", "Size");

    model.result().table("evl3").label("Evaluation 3D");
    model.result().table("evl3").comments("Interactive 3D values");

    model.component("comp1").view("view1").set("transparency", true);
    model.component("comp1").view("view1").set("clippingactive", false);
    model.component("comp1").view("view1").clip("plane1").set("translationamount", 0.0011992547040969982);
    model.component("comp1").view("view1").clip("plane1")
         .set("position", new double[]{0.05665970159680607, -0.02188580845057017, 0.005080000000000001});
    model.component("comp1").view("view1").clip("plane1")
         .set("orientationaxes", new double[][]{{0.9848077328488366, 0.17364829201905374, 0}, {-0.17364829201905374, 0.9848077328488366, 0}, {0, 0, 0.9999999999999999}});
    model.component("comp1").view("view2").axis().set("xmin", -0.11213129013776779);
    model.component("comp1").view("view2").axis().set("xmax", 0.23133867979049683);
    model.component("comp1").view("view2").axis().set("ymin", -0.1160692349076271);
    model.component("comp1").view("view2").axis().set("ymax", 0.1965576857328415);
    model.view("view3").axis().set("xmin", -0.04408358037471771);
    model.view("view3").axis().set("xmax", 0.08778329938650131);
    model.view("view3").axis().set("ymin", -0.07105730473995209);
    model.view("view3").axis().set("ymax", 0.07475701719522476);

    model.component("comp1").material("mat1").label("Air");
    model.component("comp1").material("mat1").set("family", "air");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("fununit", "Pa*s");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp")
         .set("pieces", new String[][]{{"200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("fununit", "J/(kg*K)");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("expr", "pA*0.02897/R_const[K*mol/J]/T");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("fununit", "kg/m^3");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("argunit", new String[]{"Pa", "K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("plotaxis", new String[]{"off", "on"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("plotfixedvalue", new String[]{"101325", "273.15"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("plotargs", new String[][]{{"pA", "101325", "101325"}, {"T", "273.15", "293.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("k")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("fununit", "W/(m*K)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("expr", "sqrt(1.4*R_const[K*mol/J]/0.02897*T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("fununit", "m/s");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("argunit", new String[]{"K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("plotfixedvalue", new String[]{"273.15"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("plotargs", new String[][]{{"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("funcname", "alpha_p");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("expr", "-1/rho(pA,T)*d(rho(pA,T),T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("fununit", "1/K");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("argunit", new String[]{"Pa", "K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("plotaxis", new String[]{"off", "on"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("plotfixedvalue", new String[]{"101325", "273.15"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("plotargs", new String[][]{{"pA", "101325", "101325"}, {"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("funcname", "muB");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("expr", "0.6*eta(T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("fununit", "Pa*s");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("argunit", new String[]{"K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2")
         .set("plotfixedvalue", new String[]{"200"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2")
         .set("plotargs", new String[][]{{"T", "200", "1600"}});
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)"});
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "0.02897[kg/mol]");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "muB(T)");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def").set("dynamicviscosity", "eta(T)");
    model.component("comp1").material("mat1").propertyGroup("def").set("ratioofspecificheat", "1.4");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("electricconductivity", new String[]{"0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"});
    model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", "Cp(T)");
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "rho(pA,T)");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalconductivity", new String[]{"k(T)", "0", "0", "0", "k(T)", "0", "0", "0", "k(T)"});
    model.component("comp1").material("mat1").propertyGroup("def").set("soundspeed", "cs(T)");
    model.component("comp1").material("mat1").propertyGroup("def").addInput("temperature");
    model.component("comp1").material("mat1").propertyGroup("def").addInput("pressure");
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex")
         .set("n", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("NonlinearModel").set("BA", "def.gamma-1");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").label("Piecewise 2");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp")
         .set("pieces", new String[][]{{"200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("fununit", "J/(kg*K)");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("Rs", "R_const/Mn");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("heatcapacity", "Cp(T)");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("ratioofspecificheat", "1.4");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("molarmass", "0.02897[kg/mol]");
    model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("temperature");
    model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("pressure");
    model.component("comp1").material("mat1").materialType("nonSolid");

    model.component("comp1").physics("emw").feature("port1").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port1").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port1").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port2").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port2").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port2").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port2").set("PortType", "Rectangular");

    model.component("comp1").mesh("mesh1").label("Mesh 1 40 GHz factor 7");
    model.component("comp1").mesh("mesh1").feature("ftet1").feature("size1").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("ftet1").feature("size1").set("hmax", "c_const/60[GHz]/5");
    model.component("comp1").mesh("mesh1").feature("ftet1").feature("size1").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("ftet1").feature("size1").set("hmin", 0.00346);
    model.component("comp1").mesh("mesh1").feature("ftet1").feature("size1").set("hminactive", false);
    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh2").label("Mesh 2 60 GHz factor 7");
    model.component("comp1").mesh("mesh2").feature("ftet1").feature("size1").set("custom", "on");
    model.component("comp1").mesh("mesh2").feature("ftet1").feature("size1").set("hmax", "c_const/60[GHz]/7");
    model.component("comp1").mesh("mesh2").feature("ftet1").feature("size1").set("hmaxactive", true);
    model.component("comp1").mesh("mesh2").feature("ftet1").feature("size1").set("hmin", 0.00898);
    model.component("comp1").mesh("mesh2").feature("ftet1").feature("size1").set("hminactive", false);
    model.component("comp1").mesh("mesh2").run();
    model.component("comp1").mesh("mesh3").label("Mesh 3 80 GHz factor 7");
    model.component("comp1").mesh("mesh3").feature("ftet1").feature("size1").set("custom", "on");
    model.component("comp1").mesh("mesh3").feature("ftet1").feature("size1").set("hmax", "c_const/80[GHz]/7");
    model.component("comp1").mesh("mesh3").feature("ftet1").feature("size1").set("hmaxactive", true);
    model.component("comp1").mesh("mesh3").feature("ftet1").feature("size1").set("hmin", 0.00898);
    model.component("comp1").mesh("mesh3").feature("ftet1").feature("size1").set("hminactive", false);
    model.component("comp1").mesh("mesh3").run();
    model.component("comp1").mesh("mesh4").label("Mesh 4 90 GHz factor 7");
    model.component("comp1").mesh("mesh4").feature("ftet1").feature("size1").set("custom", "on");
    model.component("comp1").mesh("mesh4").feature("ftet1").feature("size1").set("hmax", "c_const/90[GHz]/7");
    model.component("comp1").mesh("mesh4").feature("ftet1").feature("size1").set("hmaxactive", true);
    model.component("comp1").mesh("mesh4").feature("ftet1").feature("size1").set("hmin", 0.00898);
    model.component("comp1").mesh("mesh4").feature("ftet1").feature("size1").set("hminactive", false);
    model.component("comp1").mesh("mesh4").run();

    model.study().create("std1");
    model.study("std1").create("param", "Parametric");
    model.study("std1").create("freq", "Frequency");
    model.study("std1").create("freq1", "Frequency");
    model.study("std1").create("freq2", "Frequency");
    model.study("std1").create("freq3", "Frequency");

    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").attach("std1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").create("eeDef", "ErrorEstimation");
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol().create("sol2");
    model.sol("sol2").study("std1");
    model.sol("sol2").adaptationStudyFeature("std1/freq");
    model.sol("sol2").label("Adaptive Mesh Refinement Solutions 1");

    model.result().create("pg1", "PlotGroup3D");
    model.result().create("pg2", "PlotGroup1D");
    model.result().create("pg3", "SmithGroup");
    model.result("pg1").create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");
    model.result().export().create("data1", "Data");

    model.study("std1").feature("param").active(false);
    model.study("std1").feature("param").set("pname", new String[]{"r0"});
    model.study("std1").feature("param").set("plistarr", new String[]{"5 10"});
    model.study("std1").feature("param").set("punit", new String[]{"cm"});
    model.study("std1").feature("freq").label("Frequency Domain 15 - 40 GHz");
    model.study("std1").feature("freq").set("plist", "range(18,2,60)");
    model.study("std1").feature("freq").set("mesh", new String[]{"geom1", "mesh1"});
    model.study("std1").feature("freq").set("errestandadap", "errest");
    model.study("std1").feature("freq1").active(false);
    model.study("std1").feature("freq1").label("Frequency Domain 40 - 60 GHz");
    model.study("std1").feature("freq1").set("plist", "range(30,2,60)");
    model.study("std1").feature("freq1").set("mesh", new String[]{"geom1", "mesh2"});
    model.study("std1").feature("freq2").active(false);
    model.study("std1").feature("freq2").label("Frequency Domain 60 - 80 GHz");
    model.study("std1").feature("freq2").set("plist", "70 80");
    model.study("std1").feature("freq2").set("mesh", new String[]{"geom1", "mesh3"});
    model.study("std1").feature("freq3").active(false);
    model.study("std1").feature("freq3").label("Frequency Domain 80 - 90 GHz");
    model.study("std1").feature("freq3").set("plist", 90);

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain 15 - 40 GHz");
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
    model.sol("sol1").feature("s1").feature("eeDef").label("Error Estimation 1");
    model.sol("sol1").feature("s1").feature("eeDef").set("errestim", "goalerrest");
    model.sol("sol1").feature("s1").feature("eeDef").set("gfuncpdout", "comp1.field1.intop_ee(comp1.Ex)");
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
    model.result("pg1").set("looplevel", new int[]{12});
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
    model.result().export("data1").set("expr", new String[]{"emw.S21dB"});
    model.result().export("data1").set("unit", new String[]{"1"});
    model.result().export("data1").set("descr", new String[]{"S21"});
    model.result().export("data1").set("filename", "15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal");
    model.result().export("data1").set("location", "grid");
    model.result().export("data1").set("gridx3", 0);
    model.result().export("data1").set("gridy3", 0);
    model.result().export("data1").set("gridz3", 0);
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.Ez");
    model.result("pg1").feature("mslc1").set("descr", "Electric field, z-component");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").set("data", "dset1");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").setIndex("looplevel", 10, 0);
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.Ez");
    model.result("pg1").feature("mslc1").set("descr", "Electric field, z-component");
    model.result("pg1").feature("mslc1").set("data", "dset1");
    model.result("pg1").feature("mslc1").setIndex("looplevel", 5, 0);
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.normE");

    return model;
  }

  public static Model run2(Model model) {
    model.result("pg1").feature("mslc1").set("descr", "Electric field norm");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.Ez");
    model.result("pg1").feature("mslc1").set("descr", "Electric field, z-component");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", true);
    model.result("pg1").feature("mslc1").set("rangecolormin", -2.84522767829926);
    model.result("pg1").feature("mslc1").set("rangecolormax", 5);
    model.result("pg1").feature("mslc1").set("rangecolormin", -5);
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").set("data", "dset2");
    model.result("pg1").run();
    model.result("pg1").set("data", "dset1");
    model.result("pg1").setIndex("looplevel", 19, 0);
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "sqrt(emw.Ex^2+emw.Ey^2)");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", false);
    model.result("pg1").feature("mslc1").setIndex("looplevel", 12, 0);
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").setIndex("looplevel", 12, 0);
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("data", "parent");

    return model;
  }

  public static void main(String[] args) {
    Model model = run();
    run2(model);
  }

}
