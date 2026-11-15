/*
 * E4_twist_triangular.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Nov 11 2026, 10:08 by COMSOL 6.2.0.415. */
public class E4_twist_triangular {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\daniel_h\\Desktop\\wrmodes_advanced_testing\\campaign_2");

    model.label("E4_twist_triangular.mph");

    model.param().set("L", "508[mm]");
    model.param().set("FI", "90[deg]");
    model.param().set("a", "22.15[mm]");
    model.param().set("b", "47.55[mm]");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.result().table().create("tbl1", "Table");

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").create("ps1", "ParametricSurface");
    model.component("comp1").geom("geom1").feature("ps1").set("parname1", "s");
    model.component("comp1").geom("geom1").feature("ps1").set("parmax1", "L");
    model.component("comp1").geom("geom1").feature("ps1").set("parname2", "t");
    model.component("comp1").geom("geom1").feature("ps1").set("parmin2", "-b/2");
    model.component("comp1").geom("geom1").feature("ps1").set("parmax2", "b/2");
    model.component("comp1").geom("geom1").feature("ps1")
         .set("coord", new String[]{"a/2*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) - t*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "a/2*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) + t*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "s"});
    model.component("comp1").geom("geom1").feature("ps1").set("maxknots", 80);
    model.component("comp1").geom("geom1").create("ps2", "ParametricSurface");
    model.component("comp1").geom("geom1").feature("ps2").set("parname1", "s");
    model.component("comp1").geom("geom1").feature("ps2").set("parmax1", "L");
    model.component("comp1").geom("geom1").feature("ps2").set("parname2", "t");
    model.component("comp1").geom("geom1").feature("ps2").set("parmin2", "-b/2");
    model.component("comp1").geom("geom1").feature("ps2").set("parmax2", "b/2");
    model.component("comp1").geom("geom1").feature("ps2")
         .set("coord", new String[]{"-a/2*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) - t*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "-a/2*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) + t*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "s"});
    model.component("comp1").geom("geom1").feature("ps2").set("maxknots", 80);
    model.component("comp1").geom("geom1").create("ps3", "ParametricSurface");
    model.component("comp1").geom("geom1").feature("ps3").set("parname1", "s");
    model.component("comp1").geom("geom1").feature("ps3").set("parmax1", "L");
    model.component("comp1").geom("geom1").feature("ps3").set("parname2", "t");
    model.component("comp1").geom("geom1").feature("ps3").set("parmin2", "-a/2");
    model.component("comp1").geom("geom1").feature("ps3").set("parmax2", "a/2");
    model.component("comp1").geom("geom1").feature("ps3")
         .set("coord", new String[]{"t*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) - b/2*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "t*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) + b/2*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "s"});
    model.component("comp1").geom("geom1").feature("ps3").set("maxknots", 80);
    model.component("comp1").geom("geom1").create("ps4", "ParametricSurface");
    model.component("comp1").geom("geom1").feature("ps4").set("parname1", "s");
    model.component("comp1").geom("geom1").feature("ps4").set("parmax1", "L");
    model.component("comp1").geom("geom1").feature("ps4").set("parname2", "t");
    model.component("comp1").geom("geom1").feature("ps4").set("parmin2", "-a/2");
    model.component("comp1").geom("geom1").feature("ps4").set("parmax2", "a/2");
    model.component("comp1").geom("geom1").feature("ps4")
         .set("coord", new String[]{"t*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) + b/2*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "t*sin(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2))) - b/2*cos(if(s < L/2, 2*FI*(s/L)^2, FI*(1 - 2*(1 - s/L)^2)))", "s"});
    model.component("comp1").geom("geom1").feature("ps4").set("maxknots", 80);
    model.component("comp1").geom("geom1").create("wp1", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp1").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("r1").set("base", "center");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("r1").set("size", new String[]{"a", "b"});
    model.component("comp1").geom("geom1").create("wp2", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp2").set("quickz", "L");
    model.component("comp1").geom("geom1").feature("wp2").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp2").geom().create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("base", "center");
    model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("size", new String[]{"b", "a"});
    model.component("comp1").geom("geom1").create("csol1", "ConvertToSolid");
    model.component("comp1").geom("geom1").feature("csol1").set("repairtoltype", "relative");
    model.component("comp1").geom("geom1").feature("csol1").selection("input")
         .set("ps1", "ps2", "ps3", "ps4", "wp1", "wp2");
    model.component("comp1").geom("geom1").create("ext1", "Extrude");
    model.component("comp1").geom("geom1").feature("ext1").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext1").setIndex("distance", "0.15", 0);
    model.component("comp1").geom("geom1").feature("ext1").selection("inputface").set("csol1(1)", 2);
    model.component("comp1").geom("geom1").create("ext2", "Extrude");
    model.component("comp1").geom("geom1").feature("ext2").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext2").setIndex("distance", "0.15", 0);
    model.component("comp1").geom("geom1").feature("ext2").selection("inputface").set("ext1(1)", 9);
    model.component("comp1").geom("geom1").create("ext3", "Extrude");
    model.component("comp1").geom("geom1").feature("ext3").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext3").setIndex("distance", "0.02", 0);
    model.component("comp1").geom("geom1").feature("ext3").selection("inputface").set("ext2(1)", 11);
    model.component("comp1").geom("geom1").create("ext4", "Extrude");
    model.component("comp1").geom("geom1").feature("ext4").set("extrudefrom", "faces");
    model.component("comp1").geom("geom1").feature("ext4").setIndex("distance", "0.02", 0);
    model.component("comp1").geom("geom1").feature("ext4").selection("inputface").set("ext3(1)", 6);
    model.component("comp1").geom("geom1").run();

    model.component("comp1").view("view1").clip().create("plane1", "ClipPlane");

    model.component("comp1").material().create("mat1", "Common");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection().set(4);

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 2);
    model.component("comp1").physics("emw").feature("port1").selection().set(18);
    model.component("comp1").physics("emw").create("port2", "Port", 2);
    model.component("comp1").physics("emw").feature("port2").selection().set(18);
    model.component("comp1").physics("emw").create("port3", "Port", 2);
    model.component("comp1").physics("emw").feature("port3").selection().set(18);
    model.component("comp1").physics("emw").create("port4", "Port", 2);
    model.component("comp1").physics("emw").feature("port4").selection().set(18);
    model.component("comp1").physics("emw").create("port5", "Port", 2);
    model.component("comp1").physics("emw").feature("port5").selection().set(18);
    model.component("comp1").physics("emw").create("port6", "Port", 2);
    model.component("comp1").physics("emw").feature("port6").selection().set(18);
    model.component("comp1").physics("emw").create("port7", "Port", 2);
    model.component("comp1").physics("emw").feature("port7").selection().set(18);
    model.component("comp1").physics("emw").create("port8", "Port", 2);
    model.component("comp1").physics("emw").feature("port8").selection().set(18);
    model.component("comp1").physics("emw").create("port9", "Port", 2);
    model.component("comp1").physics("emw").feature("port9").selection().set(18);
    model.component("comp1").physics("emw").create("port10", "Port", 2);
    model.component("comp1").physics("emw").feature("port10").selection().set(18);
    model.component("comp1").physics("emw").create("port11", "Port", 2);
    model.component("comp1").physics("emw").feature("port11").selection().set(18);
    model.component("comp1").physics("emw").create("port12", "Port", 2);
    model.component("comp1").physics("emw").feature("port12").selection().set(18);
    model.component("comp1").physics("emw").create("port13", "Port", 2);
    model.component("comp1").physics("emw").feature("port13").selection().set(18);
    model.component("comp1").physics("emw").create("port14", "Port", 2);
    model.component("comp1").physics("emw").feature("port14").selection().set(18);
    model.component("comp1").physics("emw").create("port15", "Port", 2);
    model.component("comp1").physics("emw").feature("port15").selection().set(18);
    model.component("comp1").physics("emw").create("port16", "Port", 2);
    model.component("comp1").physics("emw").feature("port16").selection().set(18);
    model.component("comp1").physics("emw").create("port17", "Port", 2);
    model.component("comp1").physics("emw").feature("port17").selection().set(18);
    model.component("comp1").physics("emw").create("port18", "Port", 2);
    model.component("comp1").physics("emw").feature("port18").selection().set(18);
    model.component("comp1").physics("emw").create("port19", "Port", 2);
    model.component("comp1").physics("emw").feature("port19").selection().set(18);
    model.component("comp1").physics("emw").create("port20", "Port", 2);
    model.component("comp1").physics("emw").feature("port20").selection().set(18);
    model.component("comp1").physics("emw").create("port21", "Port", 2);
    model.component("comp1").physics("emw").feature("port21").selection().set(18);
    model.component("comp1").physics("emw").create("port22", "Port", 2);
    model.component("comp1").physics("emw").feature("port22").selection().set(18);
    model.component("comp1").physics("emw").create("port23", "Port", 2);
    model.component("comp1").physics("emw").feature("port23").selection().set(18);
    model.component("comp1").physics("emw").create("port24", "Port", 2);
    model.component("comp1").physics("emw").feature("port24").selection().set(8);
    model.component("comp1").physics("emw").create("port25", "Port", 2);
    model.component("comp1").physics("emw").feature("port25").selection().set(8);
    model.component("comp1").physics("emw").create("port26", "Port", 2);
    model.component("comp1").physics("emw").feature("port26").selection().set(8);
    model.component("comp1").physics("emw").create("port27", "Port", 2);
    model.component("comp1").physics("emw").feature("port27").selection().set(8);
    model.component("comp1").physics("emw").create("port28", "Port", 2);
    model.component("comp1").physics("emw").feature("port28").selection().set(8);
    model.component("comp1").physics("emw").create("port29", "Port", 2);
    model.component("comp1").physics("emw").feature("port29").selection().set(8);
    model.component("comp1").physics("emw").create("port30", "Port", 2);
    model.component("comp1").physics("emw").feature("port30").selection().set(8);
    model.component("comp1").physics("emw").create("port31", "Port", 2);
    model.component("comp1").physics("emw").feature("port31").selection().set(8);
    model.component("comp1").physics("emw").create("port32", "Port", 2);
    model.component("comp1").physics("emw").feature("port32").selection().set(8);
    model.component("comp1").physics("emw").create("port33", "Port", 2);
    model.component("comp1").physics("emw").feature("port33").selection().set(8);
    model.component("comp1").physics("emw").create("port34", "Port", 2);
    model.component("comp1").physics("emw").feature("port34").selection().set(8);
    model.component("comp1").physics("emw").create("port35", "Port", 2);
    model.component("comp1").physics("emw").feature("port35").selection().set(8);
    model.component("comp1").physics("emw").create("port36", "Port", 2);
    model.component("comp1").physics("emw").feature("port36").selection().set(8);
    model.component("comp1").physics("emw").create("port37", "Port", 2);
    model.component("comp1").physics("emw").feature("port37").selection().set(8);
    model.component("comp1").physics("emw").create("port38", "Port", 2);
    model.component("comp1").physics("emw").feature("port38").selection().set(8);
    model.component("comp1").physics("emw").create("port39", "Port", 2);
    model.component("comp1").physics("emw").feature("port39").selection().set(8);
    model.component("comp1").physics("emw").create("port40", "Port", 2);
    model.component("comp1").physics("emw").feature("port40").selection().set(8);
    model.component("comp1").physics("emw").create("port41", "Port", 2);
    model.component("comp1").physics("emw").feature("port41").selection().set(8);
    model.component("comp1").physics("emw").create("port42", "Port", 2);
    model.component("comp1").physics("emw").feature("port42").selection().set(8);
    model.component("comp1").physics("emw").create("port43", "Port", 2);
    model.component("comp1").physics("emw").feature("port43").selection().set(8);
    model.component("comp1").physics("emw").create("port44", "Port", 2);
    model.component("comp1").physics("emw").feature("port44").selection().set(8);
    model.component("comp1").physics("emw").create("port45", "Port", 2);
    model.component("comp1").physics("emw").feature("port45").selection().set(8);
    model.component("comp1").physics("emw").create("port46", "Port", 2);
    model.component("comp1").physics("emw").feature("port46").selection().set(8);

    model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");

    model.result().table("tbl1").comments("Global Evaluation 1");

    model.component("comp1").view("view1").set("renderwireframe", true);
    model.component("comp1").view("view1").set("clippingactive", false);
    model.component("comp1").view("view1").clip("plane1").set("translationamount", 0.006262695275497013);
    model.component("comp1").view("view1").clip("plane1")
         .set("position", new double[]{-6.938893903907228E-18, 6.938893903907228E-18, 0.254});
    model.component("comp1").view("view1").clip("plane1")
         .set("orientationaxes", new double[][]{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}});
    model.component("comp1").view("view2").axis().set("xmin", -0.03985155001282692);
    model.component("comp1").view("view2").axis().set("xmax", 0.03985228016972542);
    model.component("comp1").view("view2").axis().set("ymin", -0.028843319043517113);
    model.component("comp1").view("view2").axis().set("ymax", 0.02884404920041561);
    model.component("comp1").view("view3").axis().set("xmin", -0.03985155001282692);
    model.component("comp1").view("view3").axis().set("xmax", 0.03985228016972542);
    model.component("comp1").view("view3").axis().set("ymin", -0.028843319043517113);
    model.component("comp1").view("view3").axis().set("ymax", 0.02884404920041561);

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

    model.component("comp1").physics("emw").feature("port1").set("PortExcitation", "off");
    model.component("comp1").physics("emw").feature("port1").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port1").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port1").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port2").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port2").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port2").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port2").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port2").set("PortModeNumber", "20");
    model.component("comp1").physics("emw").feature("port3").set("PortExcitation", "on");
    model.component("comp1").physics("emw").feature("port3").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port3").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port3").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port3").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port3").set("PortModeNumber", "01");
    model.component("comp1").physics("emw").feature("port4").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port4").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port4").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port4").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port4").set("PortModeNumber", "11");
    model.component("comp1").physics("emw").feature("port5").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port5").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port5").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port5").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port5").set("PortModeNumber", "11");
    model.component("comp1").physics("emw").feature("port5").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port6").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port6").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port6").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port6").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port6").set("PortModeNumber", "21");
    model.component("comp1").physics("emw").feature("port7").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port7").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port7").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port7").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port7").set("PortModeNumber", "21");
    model.component("comp1").physics("emw").feature("port7").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port8").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port8").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port8").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port8").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port8").set("PortModeNumber", "30");
    model.component("comp1").physics("emw").feature("port9").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port9").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port9").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port9").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port9").set("PortModeNumber", "31");
    model.component("comp1").physics("emw").feature("port10").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port10").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port10").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port10").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port10").set("PortModeNumber", "31");
    model.component("comp1").physics("emw").feature("port10").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port11").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port11").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port11").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port11").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port11").set("PortModeNumber", "40");
    model.component("comp1").physics("emw").feature("port12").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port12").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port12").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port12").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port12").set("PortModeNumber", "02");
    model.component("comp1").physics("emw").feature("port13").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port13").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port13").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port13").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port13").set("PortModeNumber", "12");
    model.component("comp1").physics("emw").feature("port14").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port14").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port14").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port14").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port14").set("PortModeNumber", "12");
    model.component("comp1").physics("emw").feature("port14").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port15").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port15").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port15").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port15").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port15").set("PortModeNumber", "41");
    model.component("comp1").physics("emw").feature("port16").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port16").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port16").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port16").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port16").set("PortModeNumber", "41");
    model.component("comp1").physics("emw").feature("port16").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port17").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port17").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port17").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port17").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port17").set("PortModeNumber", "22");
    model.component("comp1").physics("emw").feature("port18").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port18").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port18").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port18").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port18").set("PortModeNumber", "22");
    model.component("comp1").physics("emw").feature("port18").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port19").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port19").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port19").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port19").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port19").set("PortModeNumber", "50");
    model.component("comp1").physics("emw").feature("port20").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port20").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port20").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port20").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port20").set("PortModeNumber", "32");
    model.component("comp1").physics("emw").feature("port21").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port21").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port21").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port21").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port21").set("PortModeNumber", "32");
    model.component("comp1").physics("emw").feature("port21").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port22").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port22").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port22").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port22").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port22").set("PortModeNumber", "51");
    model.component("comp1").physics("emw").feature("port23").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port23").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port23").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port23").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port23").set("PortModeNumber", "51");
    model.component("comp1").physics("emw").feature("port23").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port24").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port24").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port24").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port24").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port25").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port25").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port25").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port25").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port25").set("PortModeNumber", "20");
    model.component("comp1").physics("emw").feature("port26").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port26").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port26").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port26").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port26").set("PortModeNumber", "01");
    model.component("comp1").physics("emw").feature("port27").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port27").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port27").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port27").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port27").set("PortModeNumber", "11");
    model.component("comp1").physics("emw").feature("port28").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port28").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port28").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port28").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port28").set("PortModeNumber", "11");
    model.component("comp1").physics("emw").feature("port28").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port29").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port29").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port29").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port29").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port29").set("PortModeNumber", "21");
    model.component("comp1").physics("emw").feature("port30").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port30").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port30").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port30").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port30").set("PortModeNumber", "21");
    model.component("comp1").physics("emw").feature("port30").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port31").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port31").set("SlitType", "DomainBacked");

    return model;
  }

  public static Model run2(Model model) {
    model.component("comp1").physics("emw").feature("port31").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port31").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port31").set("PortModeNumber", "30");
    model.component("comp1").physics("emw").feature("port32").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port32").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port32").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port32").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port32").set("PortModeNumber", "31");
    model.component("comp1").physics("emw").feature("port33").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port33").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port33").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port33").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port33").set("PortModeNumber", "31");
    model.component("comp1").physics("emw").feature("port33").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port34").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port34").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port34").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port34").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port34").set("PortModeNumber", "40");
    model.component("comp1").physics("emw").feature("port35").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port35").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port35").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port35").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port35").set("PortModeNumber", "02");
    model.component("comp1").physics("emw").feature("port36").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port36").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port36").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port36").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port36").set("PortModeNumber", "12");
    model.component("comp1").physics("emw").feature("port37").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port37").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port37").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port37").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port37").set("PortModeNumber", "12");
    model.component("comp1").physics("emw").feature("port37").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port38").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port38").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port38").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port38").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port38").set("PortModeNumber", "41");
    model.component("comp1").physics("emw").feature("port39").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port39").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port39").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port39").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port39").set("PortModeNumber", "41");
    model.component("comp1").physics("emw").feature("port39").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port40").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port40").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port40").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port40").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port40").set("PortModeNumber", "22");
    model.component("comp1").physics("emw").feature("port41").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port41").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port41").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port41").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port41").set("PortModeNumber", "22");
    model.component("comp1").physics("emw").feature("port41").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port42").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port42").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port42").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port42").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port42").set("PortModeNumber", "50");
    model.component("comp1").physics("emw").feature("port43").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port43").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port43").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port43").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port43").set("PortModeNumber", "32");
    model.component("comp1").physics("emw").feature("port44").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port44").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port44").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port44").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port44").set("PortModeNumber", "32");
    model.component("comp1").physics("emw").feature("port44").set("PortModeType", "TM");
    model.component("comp1").physics("emw").feature("port45").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port45").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port45").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port45").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port45").set("PortModeNumber", "51");
    model.component("comp1").physics("emw").feature("port46").set("PortSlit", true);
    model.component("comp1").physics("emw").feature("port46").set("SlitType", "DomainBacked");
    model.component("comp1").physics("emw").feature("port46").set("PortOrientation", "ReversePort");
    model.component("comp1").physics("emw").feature("port46").set("PortType", "Rectangular");
    model.component("comp1").physics("emw").feature("port46").set("PortModeNumber", "51");
    model.component("comp1").physics("emw").feature("port46").set("PortModeType", "TM");

    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", "c_const/18[GHz]/5");
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "c_const/18[GHz]/50");
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
    model.result("pg1").feature("mslc1").create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result("pg2").create("glob1", "Global");
    model.result("pg3").create("rgr1", "ReflectionGraph");
    model.result("pg3").feature("rgr1").create("col1", "Color");
    model.result("pg3").feature("rgr1").feature("col1").set("expr", "emw.freq/1e9");

    model.study("std1").feature("freq").set("plist", "15 16 17 18");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Frequency Domain");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"15[GHz] 16[GHz] 17[GHz] 18[GHz]"});
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").set("probesel", "none");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 1");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("aDef").set("cachepattern", true);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").label("Parametric 1.1");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"15 16 17 18"});
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
    model.result().numerical("gev1")
         .set("expr", new String[]{"emw.S10_3", "emw.S11_3", "emw.S12_3", "emw.S13", "emw.S13_3", "emw.S14_3", "emw.S15_3", "emw.S16_3", "emw.S17_3", "emw.S18_3", 
         "emw.S19_3", "emw.S20_3", "emw.S21_3", "emw.S22_3", "emw.S23", "emw.S23_3", "emw.S24_3", "emw.S25_3", "emw.S26_3", "emw.S27_3", 
         "emw.S28_3", "emw.S29_3", "emw.S30_3", "emw.S31_3", "emw.S32_3", "emw.S33", "emw.S33_3", "emw.S34_3", "emw.S35_3", "emw.S36_3", 
         "emw.S37_3", "emw.S38_3", "emw.S39_3", "emw.S40_3", "emw.S41_3", "emw.S42_3", "emw.S43", "emw.S43_3", "emw.S44_3", "emw.S45_3", 
         "emw.S46_3", "emw.S53", "emw.S63", "emw.S73", "emw.S83", "emw.S93"});
    model.result().numerical("gev1")
         .set("unit", new String[]{"1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1"});
    model.result().numerical("gev1")
         .set("descr", new String[]{"S10_3", "S11_3", "S12_3", "S13", "S13_3", "S14_3", "S15_3", "S16_3", "S17_3", "S18_3", 
         "S19_3", "S20_3", "S21_3", "S22_3", "S23", "S23_3", "S24_3", "S25_3", "S26_3", "S27_3", 
         "S28_3", "S29_3", "S30_3", "S31_3", "S32_3", "S33", "S33_3", "S34_3", "S35_3", "S36_3", 
         "S37_3", "S38_3", "S39_3", "S40_3", "S41_3", "S42_3", "S43", "S43_3", "S44_3", "S45_3", 
         "S46_3", "S53", "S63", "S73", "S83", "S93"});
    model.result().numerical("gev1").setResult();
    model.result("pg1").label("Electric Field (emw)");
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
    model.result("pg2").feature("glob1")
         .set("expr", new String[]{"emw.S13dB", "emw.S10_3dB", "emw.S11_3dB", "emw.S12_3dB", "emw.S13_3dB", "emw.S14_3dB", "emw.S15_3dB", "emw.S16_3dB", "emw.S17_3dB", "emw.S18_3dB", 
         "emw.S19_3dB", "emw.S23dB", "emw.S20_3dB", "emw.S21_3dB", "emw.S22_3dB", "emw.S23_3dB", "emw.S24_3dB", "emw.S25_3dB", "emw.S26_3dB", "emw.S27_3dB", 
         "emw.S28_3dB", "emw.S29_3dB", "emw.S33dB", "emw.S30_3dB", "emw.S31_3dB", "emw.S32_3dB", "emw.S33_3dB", "emw.S34_3dB", "emw.S35_3dB", "emw.S36_3dB", 
         "emw.S37_3dB", "emw.S38_3dB", "emw.S39_3dB", "emw.S43dB", "emw.S40_3dB", "emw.S41_3dB", "emw.S42_3dB", "emw.S43_3dB", "emw.S44_3dB", "emw.S45_3dB", 
         "emw.S46_3dB", "emw.S53dB", "emw.S63dB", "emw.S73dB", "emw.S83dB", "emw.S93dB"});
    model.result("pg2").feature("glob1")
         .set("unit", new String[]{"1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
         "1", "1", "1", "1", "1", "1"});
    model.result("pg2").feature("glob1")
         .set("descr", new String[]{"S13", "S10_3", "S11_3", "S12_3", "S13_3", "S14_3", "S15_3", "S16_3", "S17_3", "S18_3", 
         "S19_3", "S23", "S20_3", "S21_3", "S22_3", "S23_3", "S24_3", "S25_3", "S26_3", "S27_3", 
         "S28_3", "S29_3", "S33", "S30_3", "S31_3", "S32_3", "S33_3", "S34_3", "S35_3", "S36_3", 
         "S37_3", "S38_3", "S39_3", "S43", "S40_3", "S41_3", "S42_3", "S43_3", "S44_3", "S45_3", 
         "S46_3", "S53", "S63", "S73", "S83", "S93"});
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
    Model model = run();
    run2(model);
  }

}
