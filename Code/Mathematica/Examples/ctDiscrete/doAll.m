(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



(* This notebook solves and simulates the TractableBufferStock model *)


(* This cell is uninteresting housekeeping and setup stuff; it can be ignored *)
ClearAll["Global`*"];ParamsAreSet=False;
If[$VersionNumber<8,(*then*) Print["These programs require Mathematica version 8 or greater."];Abort[]];
If[Length[$FrontEnd] > 0,NBDir=SetDirectory[NotebookDirectory[]]];(* If running from the Notebook interface *)
rootDir = SetDirectory["../../.."];
AutoLoadDir=SetDirectory["./Mathematica/CoreCode/Autoload"];Get["./init.m"];
CoreCodeDir=SetDirectory[".."];
rootDir = SetDirectory[".."];
Get[CoreCodeDir<>"/MakeAnalyticalResults.m"];
Get[CoreCodeDir<>"/VarsAndFuncs.m"];
(* Method of creating figures depends on whether being run in batch mode or interactively *)
If[$FrontEnd == Null,OpenFigsUsingShell=True,OpenFigsUsingShell=False]; 
FigsDir=SetDirectory[rootDir<>"/Examples/ctDiscrete/Figures/"];
CodeDir=SetDirectory[rootDir<>"/Examples/ctDiscrete"];


Get[CoreCodeDir<>"/ParametersBase.m"];
FindStableArm;\[Kappa]EBase = \[Kappa]E;


StableArmStyle={Black,Dashing[{.01}],Thickness[Medium]};
Get[CodeDir<>"/cFunc.m"];
ExportFigsToDir["TractableBufferStockcFunc",FigsDir];


Get[CodeDir <> "/PhaseDiag.m"];
ExportFigsToDir["TractableBufferStockPhaseDiag",FigsDir];



Get[CoreCodeDir<>"/ParametersBase.m"];
FindStableArm;\[Kappa]EBase = \[Kappa]E;
HorizAxis=((r)-\[CurlyTheta])/\[Rho]-0.01;
\[ScriptC]GroMaxPlot=Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.5` \[ScriptM]E]];
BufferFigPlot=Show[Plot[Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5` \[ScriptM]E,2.5` \[ScriptM]E}
,Ticks->{{{\[ScriptM]E,"\!\(\*SuperscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(e\)]\)"}},{{\[GothicG]+\[Mho],"\[Gamma]"},{((r)-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(r-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]}}}
,PlotRange->{{0,2.5` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
(*,Graphics[{Dashing[{0.005`,0.025`}],Thickness[Medium],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]*)
,Graphics[{Dashing[{}],Thickness[Small],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,\[GothicG]+\[Mho]},{2.5` \[ScriptM]E,\[GothicG]+\[Mho]}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.5` \[ScriptM]E,((r)-\[CurlyTheta])/\[Rho]}}]}
,PlotRange->{{0,2.5` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
,AxesOrigin->{((r)-\[CurlyTheta])/\[Rho]-0.1,HorizAxis}
];
(* The thorn character \[Thorn] does not appear properly unless encoded with the WindowsANSI encoding, which necessitates the cumbersome apparatus below *)
cLev=Style["c",{Bold,Italic},CharacterEncoding->"WindowsANSI"];
cELevtp1=SubsuperscriptBox[cLev,Style["t+1",CharacterEncoding->"WindowsANSI"],Style["e",CharacterEncoding->"WindowsANSI"]] //DisplayForm;
dLog = Style["\!\(\*
StyleBox[\"\[CapitalDelta]\",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\"log\",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Plain\"]\)",CharacterEncoding->"WindowsANSI"];
ArrowPointingLeft = Style[" \[LongLeftArrow] ",CharacterEncoding->"WindowsANSI"];
ArrowPointingRight = Style[" \[LongRightArrow] ",CharacterEncoding->"WindowsANSI"];
ApproxcELevGro = Style[" \[TildeTilde] \[Thorn] + \[Mho](1+\[Omega]\!\(\*SubscriptBox[\(\[Del]\), \(t + 1\)]\))\!\(\*SubscriptBox[\(\[Del]\), \(t + 1\)]\)",CharacterEncoding->"WindowsANSI"];
TractableBufferStockGrowthA=BufferFigBaseline=Show[BufferFigPlot
,Graphics[Text[DisplayForm[RowBox[{ArrowPointingLeft,dLog,cELevtp1,ApproxcELevGro}]],{(\[ScriptM]E 2)/3,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]E 2)/3]]},{-1,0}]]
,Graphics[Text[" \!\(\*
StyleBox[\"{\",\nFontSize->36]\)",{\[ScriptM]E 2,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])},{1,0}]]
,Graphics[Text["Precautionary Increment: \[Mho](1+\[Omega]\!\(\*SubscriptBox[\(\[Del]\), \(t + 1\)]\))\!\(\*SubscriptBox[\(\[Del]\), \(t + 1\)]\)   ",{\[ScriptM]E 2,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])},{1,0}]]
,Axes->{Automatic,Automatic}
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)","Growth"}
,AxesOrigin->{((r)-\[CurlyTheta])/\[Rho]-0.1,HorizAxis}
,PlotRange->{{((r)-\[CurlyTheta])/\[Rho]-0.1,2.5` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}];
ExportFigsToDir["TractableBufferStockGrowthA",FigsDir];
TractableBufferStockGrowthA;


 r = rBase+.04;
FindStableArm;
cEModLevtp1=SubsuperscriptBox[OverscriptBox[Style["c",{Bold,Italic},CharacterEncoding->"WindowsANSI"],"`"],Style["t+1",Plain],Style["e",Plain]];
BufferFigNew = Plot[
Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5 \[ScriptM]EBase,2.5 \[ScriptM]EBase}
,PlotStyle->{Dashing[{.01}],Thickness[Medium],Black}];
TractableBufferStockGrowthB=Show[BufferFigPlot
,BufferFigNew
,Graphics[Text[DisplayForm[RowBox[{ArrowPointingLeft,dLog,cEModLevtp1}]],{(\[ScriptM]EBase 14)/8,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 13)/8]]},{-1,0}]]
,Graphics[Text[DisplayForm[RowBox[{dLog, cELevtp1, ArrowPointingRight,"   "}]],{(\[ScriptM]EBase 5)/6,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 5)/6]]},{1,1}]]
,Graphics[{Dashing[{0.005`,0.025`}],Thickness[Medium],Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[1.5`]]}}]}]
,Graphics[{Dashing[{0.01`}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.5` \[ScriptM]EBase,((r)-\[CurlyTheta])/\[Rho]}}]}]
,PhaseArrow[{\[ScriptM]E 1.25`,(rBase-\[CurlyTheta])/\[Rho]},{\[ScriptM]E 1.25`,((r)-\[CurlyTheta])/\[Rho]}]
,PhaseArrow[{\[ScriptM]E 1.25`,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 1.25`]]-(((r)-rBase) 2)/(\[Rho] 4)},{\[ScriptM]E 1.25`,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 1.25`]]}]
,Axes->{Automatic,Automatic}
,AxesLabel->{"\!\(\*SubscriptBox[\(\[ScriptM]\), \(\[ScriptT]\)]\)","Growth"}
,AxesOrigin->{Automatic,HorizAxis}
,Ticks->{{{\[ScriptM]EBase,"\!\(\*OverscriptBox[\(m\), \(\[Hacek]\)]\)"},{\[ScriptM]E,"\!\(\*OverscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(`\)]\)"}},{{\[GothicG]+\[Mho],"\[Gamma]"},{(rBase-\[CurlyTheta])/\[Rho],
Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(\[ScriptR]-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]},{((r)-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(\!\(\*OverscriptBox[\(\[ScriptR]\), \(`\)]\)-\[CurlyTheta])\[TildeTilde]\!\(\*OverscriptBox[\(\[Thorn]\), \(`\)]\)",CharacterEncoding->"WindowsANSI"]}}}
,PlotRange->{{0,1.8` \[ScriptM]E},{HorizAxis,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.5`\[ScriptM]E]]}}];
ExportFigsToDir["TractableBufferStockGrowthB",FigsDir];
TractableBufferStockGrowthB;


  r = rBase;
\[CurlyTheta]=\[CurlyTheta]Base-0.02;
FindStableArm;
\[ScriptM]Max0=\[ScriptM]Max;
cFuncPlotNew=Plot[cE[\[ScriptM]],{\[ScriptM],0,\[ScriptM]Max0},PlotStyle->{Black,Thickness[Medium]}];
cFuncPlotNewPoints=Map[{{#,\[ScriptC][#]}}&,Table[\[ScriptM],{\[ScriptM],0,\[ScriptM]Max0,0.01}]];
Stable\[ScriptM]LocusPlot=Plot[{\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,\[ScriptM]Max0},PlotStyle->{Dashing[{.01}],Thickness[Medium],Black}];
Stable\[ScriptM]LocusPoints=Map[{{#,\[ScriptM]EDelEqZero[#]}}&,Table[\[ScriptM],{\[ScriptM],0,\[ScriptM]Max0,0.1}]];


SetOptions[ListPlot,PlotStyle->Black];SimGeneratePath[\[ScriptM]EBase,200];
\[ScriptM]\[ScriptC]PathPlot = ListPlot[\[ScriptM]\[ScriptC]Path,PlotStyle->{Black,PointSize[0.008]}];
{\[ScriptM]MinNew,\[ScriptM]MaxNew}={0,2 \[ScriptM]EBase};
{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}={0,1.5 \[ScriptC]EBase};
TractableBufferStockTarget = Show[cFuncPlotBase,Stable\[ScriptM]LocusPlot
,Graphics[Text["Target \[LongRightArrow]",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,0}]]
,Graphics[Text["     \[LongLeftArrow] Sustainable \[ScriptC]",{1.3 \[ScriptM]EBase,\[ScriptM]EDelEqZero[1.3 \[ScriptM]EBase]},{-1,0}]]
,Graphics[Text["c(\[ScriptM]) \[LongRightArrow]   ",{0.3\[ScriptM]EBase,0.5\[ScriptC]EBase},{1,0}]]
,Ticks->None
,PlotRange->{{\[ScriptM]MinNew,\[ScriptM]Max},{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}];
OldAndNewcFuncsPlot = Show[Stable\[ScriptM]LocusPlot,cFuncPlotBase,cFuncPlotNew
,PlotRange->{{0,\[ScriptM]Max},{0,Automatic}}
,AxesOrigin->{0.,0.}
];
ExportFigsToDir["TractableBufferStockTarget",FigsDir];Show[TractableBufferStockTarget]


PhaseDiagramDecreaseThetaPlot = Show[OldAndNewcFuncsPlot,\[ScriptM]\[ScriptC]PathPlot
,Graphics[Text["Original Target \[LowerRightArrow] ",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,-1}]]
,Graphics[Text["\[UpperLeftArrow] New Target ",{\[ScriptM]E,0.98\[ScriptC]E},{-1,1}]]
,Graphics[Text["Original c(\[ScriptM]) \[LongRightArrow]",{0.45\[ScriptM]EBase,0.67\[ScriptC]EBase},{1,0}]]
,Graphics[Text[" \[LongLeftArrow] New c(\[ScriptM])",{0.8\[ScriptM]EBase,cE[0.8\[ScriptM]EBase]},{-1,0}]]
,Ticks->None
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
,PlotRange->{{\[ScriptM]MinNew-1,\[ScriptM]E+2},{\[ScriptC]MinNew,1.3\[ScriptC]E}}
,AxesOrigin->{0.,0.}
];
ExportFigsToDir["PhaseDiagramDecreaseThetaPlot",FigsDir];
Show[PhaseDiagramDecreaseThetaPlot]



HowMany=75;
\[ScriptM]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[1]],HowMany];
\[ScriptC]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[2]],HowMany];
MPCPath=Map[cE'[#]&,Rest[\[ScriptM]Path]];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
timePath=Table[i,{i,Length[\[ScriptC]Path]}];
\[ScriptC]PathPlot = ListPlot[Transpose[{timePath,\[ScriptC]Path}],PlotRange->All];
\[ScriptM]PathPlot = ListPlot[Transpose[{timePath,\[ScriptM]Path}],PlotRange->All];
MPCPathPlot = ListPlot[Transpose[{timePath,MPCPath}],PlotRange->All];


cPathAfterThetaDrop=Show[\[ScriptC]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \(\[ScriptT]\), \(e\)]\)"}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["cPathAfterThetaDrop",FigsDir];
Print[Show[cPathAfterThetaDrop]];


mPathAfterThetaDrop=Show[\[ScriptM]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)"}
,PlotRange->{{-3,HowMany},{0,Automatic}}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["mPathAfterThetaDrop",FigsDir];
Print[Show[mPathAfterThetaDrop]];




MPCPathAfterThetaDrop=Show[MPCPathPlot
,Graphics[{Dashing[{0.01}],Line[{{timePath[[1]],\[Kappa]},{timePath[[-1]],\[Kappa]}}]}]
,Graphics[Text["\[UpArrow]",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa]},{0,1}]]
,Graphics[Text["Perfect Foresight MPC",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa](4/5)},{0,1}]]
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubscriptBox[\(\[Kappa]\), \(\[ScriptT]\)]\)"}
,PlotRange->All
,AxesOrigin->{-3,0.}];
ExportFigsToDir["MPCPathAfterThetaDrop",FigsDir];


Get[CoreCodeDir<>"/ParametersBase.m"];
FindStableArm;\[Kappa]EBase = \[Kappa]E;\[ScriptM]EBase=\[ScriptM]E;\[ScriptC]EBase=\[ScriptC]E;


{mMaxPlot,mMaxPlot}={1.5,5} \[ScriptM]E;
\[ScriptC]LowerPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Dashing[{.01}]];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Dashing[{.02}]];
Degree45 = Plot[\[ScriptM],{\[ScriptM],0,cE[mMaxPlot]},PlotStyle->Dashing[{.01}]];
cFuncPlotBase=cFuncPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Black];
BufferFigOrig=Show[Plot[Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5` \[ScriptM]E,2.1` \[ScriptM]E}
,Ticks->{{{\[ScriptM]E,"\!\(\*SuperscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(e\)]\)"}},{{\[GothicG]+\[Mho],"\[Gamma]"},{((r)-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(r-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]}}}
,PlotRange->{{0,2.1` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
(*,Graphics[{Dashing[{0.005`,0.025`}],Thickness[Medium],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]*)
,Graphics[{Dashing[{}],Thickness[Small],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,\[GothicG]+\[Mho]},{2.1` \[ScriptM]E,\[GothicG]+\[Mho]}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.1` \[ScriptM]E,((r)-\[CurlyTheta])/\[Rho]}}]}
,PlotRange->{{0,2.1` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
,AxesOrigin->{((r)-\[CurlyTheta])/\[Rho]-0.1,HorizAxis}
];



Stable\[ScriptM]LocusOrigPlot=Plot[{\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,mMaxPlot},PlotStyle->{Black,Dashing[{.01}],Thickness[Medium]}];

\[Mho]=3 \[Mho]Base;
FindStableArm;
cFuncPlotNew=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->RGBColor[0,0,0]];
cFuncPlotNewPoints=Map[{{#,\[ScriptC][#]}}&,Table[\[ScriptM],{\[ScriptM],0,mMaxPlot,0.01}]];
Stable\[ScriptM]LocusPlot=Plot[{\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,mMaxPlot},PlotStyle->{Black,Dashing[{.01}],Thickness[Medium]}];
Stable\[ScriptM]LocusPoints=Map[{{#,\[ScriptM]EDelEqZero[#]}}&,Table[\[ScriptM],{\[ScriptM],0,mMaxPlot,0.1}]];


SimGeneratePath[\[ScriptM]EBase,100];
\[ScriptM]\[ScriptC]PathPlot = ListPlot[\[ScriptM]\[ScriptC]Path,PlotStyle->{Black,PointSize[0.007]}];
{\[ScriptM]MinNew,\[ScriptM]Max}={0,5};
{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}={0,1.5};
TractableBufferStockTarget = Show[cFuncPlotBase,Stable\[ScriptM]LocusPlot,Stable\[ScriptM]LocusOrigPlot
,Graphics[Text["Target \[LongRightArrow]",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,0}]]
,Graphics[Text[" \[LongLeftArrow] Sustainable \[ScriptC]",{\[ScriptM]E,0.98\[ScriptC]E},{-1,0}]]
,Graphics[Text["c(\[ScriptM]) \[LongRightArrow]",{0.7\[ScriptM]EBase,0.85\[ScriptC]EBase},{1,0}]]
,Ticks->None
,PlotRange->{{\[ScriptM]MinNew,\[ScriptM]Max},{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}];
OldAndNewcFuncsPlot = Show[Stable\[ScriptM]LocusPlot,cFuncPlotBase,cFuncPlotNew
,PlotRange->{{0,mMaxPlot},{0,Automatic}}
,AxesOrigin->{0.,0.}
];



PhaseDiagramIncreaseMhoPlot = Show[OldAndNewcFuncsPlot,\[ScriptM]\[ScriptC]PathPlot,Stable\[ScriptM]LocusOrigPlot
,Graphics[Text["Orig Target \[LongRightArrow]",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,0}]]
,Graphics[Text["\[UpperLeftArrow]New Target",{\[ScriptM]E,0.96\[ScriptC]E},{-1,0}]]
,Graphics[Text["Orig c(\[ScriptM]) \[LongRightArrow]",{1.3\[ScriptM]EBase,1.2\[ScriptC]EBase},{1,0}]]
,Graphics[Text[" \[LongLeftArrow] New c(\[ScriptM])",{\[ScriptM]EBase,cE[\[ScriptM]EBase]},{-1,0}]]
,Ticks->None
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
,PlotRange->{{\[ScriptM]MinNew-1,\[ScriptM]E+3},{\[ScriptC]MinNew,1.3\[ScriptC]EBase}}
,AxesOrigin->{0.,0.}
];
ExportFigsToDir["PhaseDiagramIncreaseMhoPlot",FigsDir];



cEModLevtp1=SubsuperscriptBox[OverscriptBox[Style["c",{Bold,Italic},CharacterEncoding->"WindowsANSI"],"`"],Style["t+1",Plain],Style["e",Plain]];
BufferFigNew = Plot[
Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5 \[ScriptM]EBase,2.1 \[ScriptM]EBase}
,PlotStyle->{Dashing[{.01}],Thickness[Medium],Black}];
cGroIncreaseMhoPlot=Show[BufferFigOrig
,BufferFigNew
(*,Graphics[Text[DisplayForm[RowBox[{ArrowPointingLeft,dLog,cEModLevtp1}]],{(\[ScriptM]EBase 14)/8,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 13)/8]]},{-1,0}]]*)
,Graphics[Text[DisplayForm[RowBox[{dLog, cEModLevtp1,ArrowPointingRight}]],{(\[ScriptM]EBase 7.5)/12,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 7.5)/12]]},{1,1}]]
,Graphics[{Dashing[{}],Thickness[Small],Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[1.5`]]}}]}]
(*,Graphics[{Dashing[{0.01`}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.0` \[ScriptM]EBase,((r)-\[CurlyTheta])/\[Rho]}}]}]*)
,Graphics[{Dashing[{}],Thickness[Small],Black,Line[{{\[ScriptM]EBase,HorizAxis},{\[ScriptM]EBase,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[1.5`]]}}]}]
,Graphics[{Dashing[{0.01`}],Thickness[Medium],Line[{{0,\[GothicG]+\[Mho]},{2.1` \[ScriptM]EBase,\[GothicG]+\[Mho]}}]}]
(*,Graphics[Text[" {",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])-0.003},{1,0}]]*)
(*,Graphics[Text["\[Mho](1+\[Omega]Subscript[\[Del], t+1])Subscript[\[Del], t+1]   ",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])-0.003},{1,0}]]*)
(*,Graphics[Text[" }",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])+0.001},{-1,0}]]*)
(*,Graphics[Text["Overscript[\[Mho], `](1+\[Omega]Subscript[Overscript[\[Del], `], t+1])Subscript[Overscript[\[Del], `], t+1]",{\[ScriptM]E 1.85,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])+0.001},{-1,0}]]*)
,Graphics[Text["Original Eqbm \[UpperRightArrow]",{\[ScriptM]EBase,\[GothicG]+\[Mho]Base},{1,1}]]
,Graphics[Text[" \[LowerLeftArrow] New Target",{\[ScriptM]E,\[GothicG]+\[Mho]},{-1,-1}]]
,Graphics[{PointSize[0.015],Point[{\[ScriptM]EBase,\[GothicG]+\[Mho]Base}]}]
,Graphics[{PointSize[0.015],Point[{\[ScriptM]E,\[GothicG]+\[Mho]}]}]
,PhaseArrow[{0.1 \[ScriptM]EBase,\[GothicG]+\[Mho]Base},{0.1 \[ScriptM]EBase,\[GothicG]+\[Mho]}]
,PhaseArrow[{\[ScriptM]EBase,\[GothicG]+\[Mho]Base},{\[ScriptM]EBase,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]EBase]]}]
,Axes->{Automatic,Automatic}
,AxesLabel->{"\!\(\*SubscriptBox[\(\[ScriptM]\), \(\[ScriptT]\)]\)","Growth"}
,AxesOrigin->{Automatic,HorizAxis}
,Ticks->{{{\[ScriptM]EBase,"\!\(\*OverscriptBox[\(m\), \(\[Hacek]\)]\)"}
,{\[ScriptM]E,"\!\(\*OverscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(`\)]\)"}}
,{{\[GothicG]+\[Mho]Base,"\[Gamma]"}
,{\[GothicG]+\[Mho],"\!\(\*OverscriptBox[\(\[Gamma]\), \(`\)]\)"}
,{(rBase-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(\[ScriptR]-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]}
(*,{((r)-\[CurlyTheta])/\[Rho],Style["\[Rho]^-1(Overscript[\[ScriptR], `]-\[CurlyTheta])\[TildeTilde]Overscript[\[Thorn], `]",CharacterEncoding\[Rule]"WindowsANSI"]}*)
}}
(*,PlotRange\[Rule]{{0,2.6` \[ScriptM]E},{HorizAxis,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.5`\[ScriptM]E]]}}*)
,PlotRange->{{0,1.8` \[ScriptM]E},{HorizAxis,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.45`\[ScriptM]E]]}}
];
ExportFigsToDir["cGroIncreaseMhoPlot",FigsDir];
cGroIncreaseMhoPlot


HowMany=75;
\[ScriptM]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[1]],HowMany];
\[ScriptC]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[2]],HowMany];
MPCPath=Map[cE'[#]&,Rest[\[ScriptM]Path]];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
timePath=Table[i,{i,Length[\[ScriptC]Path]}];
\[ScriptC]PathPlot = ListPlot[Transpose[{timePath,\[ScriptC]Path}],PlotRange->All];
\[ScriptM]PathPlot = ListPlot[Transpose[{timePath,\[ScriptM]Path}],PlotRange->All];
MPCPathPlot = ListPlot[Transpose[{timePath,MPCPath}],PlotRange->All];


cPathAfterMhoRise=Show[\[ScriptC]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \(\[ScriptT]\), \(e\)]\)"}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["cPathAfterMhoRise",FigsDir];


mPathAfterMhoRise=Show[\[ScriptM]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)"}
,PlotRange->{{-3,HowMany},{0,Automatic}}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["mPathAfterMhoRise",FigsDir];




MPCPathAfterMhoRise=Show[MPCPathPlot
,Graphics[{Dashing[{0.01}],Line[{{timePath[[1]],\[Kappa]},{timePath[[-1]],\[Kappa]}}]}]
,Graphics[Text["\[UpArrow]",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa]},{0,1}]]
,Graphics[Text["Perfect Foresight MPC",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa](4/5)},{0,1}]]
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubscriptBox[\(\[Kappa]\), \(\[ScriptT]\)]\)"}
,PlotRange->All
,AxesOrigin->{-3,0.}];
ExportFigsToDir["MPCPathAfterMhoRise",FigsDir];


(* Now decrease expected income growth *)
Get[CoreCodeDir<>"/ParametersBase.m"];
FindStableArm;\[Kappa]EBase = \[Kappa]E;\[ScriptM]EBase=\[ScriptM]E;\[ScriptC]EBase=\[ScriptC]E;


{mMaxPlot,mMaxPlot}={1.5,5} \[ScriptM]E;
\[ScriptC]LowerPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Dashing[{.01}]];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Dashing[{.02}]];
Degree45 = Plot[\[ScriptM],{\[ScriptM],0,cE[mMaxPlot]},PlotStyle->Dashing[{.01}]];
cFuncPlotBase=cFuncPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->Black];
BufferFigOrig=Show[Plot[Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5` \[ScriptM]E,2.1` \[ScriptM]E}
,Ticks->{{{\[ScriptM]E,"\!\(\*SuperscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(e\)]\)"}},{{\[GothicG]+\[Mho],"\[Gamma]"},{((r)-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(r-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]}}}
,PlotRange->{{0,2.1` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
(*,Graphics[{Dashing[{0.005`,0.025`}],Thickness[Medium],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]*)
,Graphics[{Dashing[{}],Thickness[Small],Black,Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,\[ScriptC]GroMaxPlot}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,\[GothicG]+\[Mho]},{2.1` \[ScriptM]E,\[GothicG]+\[Mho]}}]}]
,Graphics[{Dashing[{}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.1` \[ScriptM]E,((r)-\[CurlyTheta])/\[Rho]}}]}
,PlotRange->{{0,2.1` \[ScriptM]E},{HorizAxis,\[ScriptC]GroMaxPlot}}
]
,AxesOrigin->{((r)-\[CurlyTheta])/\[Rho]-0.1,HorizAxis}
];



Stable\[ScriptM]LocusOrigPlot=Plot[{\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,mMaxPlot},PlotStyle->{Black,Dashing[{.01}],Thickness[Medium]}];

\[GothicG]   =-0.02;

FindStableArm;
cFuncPlotNew=Plot[cE[\[ScriptM]],{\[ScriptM],0,mMaxPlot},PlotStyle->RGBColor[0,0,0]];
cFuncPlotNewPoints=Map[{{#,\[ScriptC][#]}}&,Table[\[ScriptM],{\[ScriptM],0,mMaxPlot,0.01}]];
Stable\[ScriptM]LocusPlot=Plot[{\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,mMaxPlot},PlotStyle->{Black,Dashing[{.01}],Thickness[Medium]}];
Stable\[ScriptM]LocusPoints=Map[{{#,\[ScriptM]EDelEqZero[#]}}&,Table[\[ScriptM],{\[ScriptM],0,mMaxPlot,0.1}]];


SimGeneratePath[\[ScriptM]EBase,100];
\[ScriptM]\[ScriptC]PathPlot = ListPlot[\[ScriptM]\[ScriptC]Path,PlotStyle->{Black,PointSize[0.007]}];
{\[ScriptM]MinNew,\[ScriptM]Max}={0,5};
{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}={0,1.5};
TractableBufferStockTarget = Show[cFuncPlotBase,Stable\[ScriptM]LocusPlot,Stable\[ScriptM]LocusOrigPlot
,Graphics[Text["Target \[LongRightArrow]",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,0}]]
,Graphics[Text[" \[LongLeftArrow] Sustainable \[ScriptC]",{\[ScriptM]E,0.98\[ScriptC]E},{-1,0}]]
,Graphics[Text["c(\[ScriptM]) \[LongRightArrow]",{0.7\[ScriptM]EBase,0.85\[ScriptC]EBase},{1,0}]]
,Ticks->None
,PlotRange->{{\[ScriptM]MinNew,\[ScriptM]Max},{\[ScriptC]MinNew,\[ScriptC]MaxPlotNew}}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}];
OldAndNewcFuncsPlot = Show[Stable\[ScriptM]LocusPlot,cFuncPlotBase,cFuncPlotNew
,PlotRange->{{0,mMaxPlot},{0,Automatic}}
,AxesOrigin->{0.,0.}
];



PhaseDiagramAfterGFallPlot = Show[OldAndNewcFuncsPlot,\[ScriptM]\[ScriptC]PathPlot,Stable\[ScriptM]LocusOrigPlot
,Graphics[Text["Orig Target \[LongRightArrow]",{\[ScriptM]EBase,1.02\[ScriptC]EBase},{1,0}]]
,Graphics[Text["\[UpperLeftArrow]New Target",{\[ScriptM]E,0.96\[ScriptC]E},{-1,0}]]
,Graphics[Text["Orig c(\[ScriptM]) \[LongRightArrow]",{1.3\[ScriptM]EBase,1.2\[ScriptC]EBase},{1,0}]]
,Graphics[Text[" \[LongLeftArrow] New c(\[ScriptM])",{\[ScriptM]EBase,cE[\[ScriptM]EBase]},{-1,0}]]
,Ticks->None
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
,PlotRange->{{\[ScriptM]MinNew-1,\[ScriptM]E+3},{\[ScriptC]MinNew,1.3\[ScriptC]EBase}}
,AxesOrigin->{0.,0.}
];
ExportFigsToDir["PhaseDiagramAfterGFallPlot",FigsDir];



cEModLevtp1=SubsuperscriptBox[OverscriptBox[Style["c",{Bold,Italic},CharacterEncoding->"WindowsANSI"],"`"],Style["t+1",Plain],Style["e",Plain]];
BufferFigNew = Plot[
Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]],{\[ScriptM],0.5 \[ScriptM]EBase,2.1 \[ScriptM]EBase}
,PlotStyle->{Dashing[{.01}],Thickness[Medium],Black}];
cGroAfterGFallPlot=Show[BufferFigOrig
,BufferFigNew
(*,Graphics[Text[DisplayForm[RowBox[{ArrowPointingLeft,dLog,cEModLevtp1}]],{(\[ScriptM]EBase 14)/8,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 13)/8]]},{-1,0}]]*)
,Graphics[Text[DisplayForm[RowBox[{dLog, cEModLevtp1,ArrowPointingRight}]],{(\[ScriptM]EBase 7.5)/12,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[(\[ScriptM]EBase 7.5)/12]]},{1,1}]]
,Graphics[{Dashing[{}],Thickness[Small],Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[1.5`]]}}]}]
(*,Graphics[{Dashing[{0.01`}],Thickness[Medium],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{2.0` \[ScriptM]EBase,((r)-\[CurlyTheta])/\[Rho]}}]}]*)
,Graphics[{Dashing[{}],Thickness[Small],Black,Line[{{\[ScriptM]EBase,HorizAxis},{\[ScriptM]EBase,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[1.5`]]}}]}]
,Graphics[{Dashing[{0.01`}],Thickness[Medium],Line[{{0,\[GothicG]+\[Mho]},{2.1` \[ScriptM]EBase,\[GothicG]+\[Mho]}}]}]
(*,Graphics[Text[" {",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])-0.003},{1,0}]]*)
(*,Graphics[Text["\[Mho](1+\[Omega]Subscript[\[Del], t+1])Subscript[\[Del], t+1]   ",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])-0.003},{1,0}]]*)
(*,Graphics[Text[" }",{\[ScriptM]E 1.65,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])+0.001},{-1,0}]]*)
(*,Graphics[Text["Overscript[\[Mho], `](1+\[Omega]Subscript[Overscript[\[Del], `], t+1])Subscript[Overscript[\[Del], `], t+1]",{\[ScriptM]E 1.85,1/2 (Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E 2]]+((r)-\[CurlyTheta])/\[Rho])+0.001},{-1,0}]]*)
,Graphics[Text["Original Eqbm \[UpperRightArrow]",{\[ScriptM]EBase,\[GothicG]Base+\[Mho]},{1,1}]]
,Graphics[Text[" \[LowerLeftArrow] New Target",{\[ScriptM]E,\[GothicG]+\[Mho]},{-1,-1}]]
,Graphics[{PointSize[0.015],Point[{\[ScriptM]EBase,\[GothicG]Base+\[Mho]}]}]
,Graphics[{PointSize[0.015],Point[{\[ScriptM]E,\[GothicG]+\[Mho]}]}]
,PhaseArrow[{0.1 \[ScriptM]EBase,\[GothicG]Base+\[Mho]},{0.1 \[ScriptM]EBase,\[GothicG]+\[Mho]}]
,PhaseArrow[{\[ScriptM]EBase,\[GothicG]Base+\[Mho]},{\[ScriptM]EBase,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]EBase]]}]
,Axes->{Automatic,Automatic}
,AxesLabel->{"\!\(\*SubscriptBox[\(\[ScriptM]\), \(\[ScriptT]\)]\)","Growth"}
,AxesOrigin->{Automatic,HorizAxis}
,Ticks->{{{\[ScriptM]EBase,"\!\(\*OverscriptBox[\(m\), \(\[Hacek]\)]\)"}
,{\[ScriptM]E,"\!\(\*OverscriptBox[OverscriptBox[\(\[ScriptM]\), \(\[Hacek]\)], \(`\)]\)"}}
,{{\[GothicG]Base+\[Mho],"\[Gamma]"}
,{\[GothicG]+\[Mho],"\!\(\*OverscriptBox[\(\[Gamma]\), \(`\)]\)"}
,{(rBase-\[CurlyTheta])/\[Rho],Style["\!\(\*SuperscriptBox[\(\[Rho]\), \(-1\)]\)(\[ScriptR]-\[CurlyTheta])\[TildeTilde]\[Thorn]",CharacterEncoding->"WindowsANSI"]}
(*,{((r)-\[CurlyTheta])/\[Rho],Style["\[Rho]^-1(Overscript[\[ScriptR], `]-\[CurlyTheta])\[TildeTilde]Overscript[\[Thorn], `]",CharacterEncoding\[Rule]"WindowsANSI"]}*)
}}
(*,PlotRange\[Rule]{{0,2.6` \[ScriptM]E},{HorizAxis,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.5`\[ScriptM]E]]}}*)
,PlotRange->{{0,1.8` \[ScriptM]E},{HorizAxis,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[0.45`\[ScriptM]E]]}}
];
ExportFigsToDir["cGroAfterGFallPlot",FigsDir];


HowMany=75;
\[ScriptM]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[1]],HowMany];
\[ScriptC]Path=Take[Transpose[\[ScriptM]\[ScriptC]Path][[2]],HowMany];
MPCPath=Map[cE'[#]&,Rest[\[ScriptM]Path]];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[\[ScriptM]Path,\[ScriptM]EBase];
PrependTo[\[ScriptC]Path,\[ScriptC]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
PrependTo[MPCPath,\[Kappa]EBase];
timePath=Table[i,{i,Length[\[ScriptC]Path]}];
\[ScriptC]PathPlot = ListPlot[Transpose[{timePath,\[ScriptC]Path}],PlotRange->All];
\[ScriptM]PathPlot = ListPlot[Transpose[{timePath,\[ScriptM]Path}],PlotRange->All];
MPCPathPlot = ListPlot[Transpose[{timePath,MPCPath}],PlotRange->All];


cPathAfterGFall=Show[\[ScriptC]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \(\[ScriptT]\), \(e\)]\)"}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["cPathAfterGFall",FigsDir];


mPathAfterGFall=Show[\[ScriptM]PathPlot
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)"}
,PlotRange->{{-3,HowMany},{0,Automatic}}
,AxesOrigin->{-3,0}
,PlotRange->{{-3,Automatic},{0,Automatic}}
];
ExportFigsToDir["mPathAfterGFall",FigsDir];




MPCPathAfterGFall=Show[MPCPathPlot
,Graphics[{Dashing[{0.01}],Line[{{timePath[[1]],\[Kappa]},{timePath[[-1]],\[Kappa]}}]}]
,Graphics[Text["\[UpArrow]",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa]},{0,1}]]
,Graphics[Text["Perfect Foresight MPC",{(timePath[[1]]+timePath[[-1]])/2,\[Kappa](4/5)},{0,1}]]
,Ticks->{{{4,"0"}},None}
,AxesLabel->{"Time","\!\(\*SubscriptBox[\(\[Kappa]\), \(\[ScriptT]\)]\)"}
,PlotRange->All
,AxesOrigin->{-3,0.}];
ExportFigsToDir["MPCPathAfterGFall",FigsDir];



