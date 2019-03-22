%%
%% This is file `scrartcl.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% scrclass.dtx  (with options: `scrartcl')
%% scrlogo.dtx 
%% 
%% Copyright (C) 1994-2001 Markus Kohm and Frank Neukam.
%% 
%% All rights reserved. For further copyright information see the file
%% LEGAL.TXT or LEGALDE.TXT.
%% 
%% This file is part of the LaTeX2e KOMA-Script bundle.
%% ----------------------------------------------------
%% 
%% The KOMA-Script bundle bases on the Script-family created by
%% Frank Neukam 1993 and the standard documents classes created
%% by the LaTeX3-Team 1994-1996.
%% ------------------------------------------------------------------------
%% 
%%  This system is distributed in the hope that it will be useful,
%%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%% 
%% 
%% IMPORTANT NOTICE:
%% 
%% For error reports in case of UNCHANGED versions see documentation.
%% 
%% Please do not request updates from us directly. Distribution is
%% done through Mail-Servers, Mailboxes and TeX organizations.
%% 
%% You are not allowed to change this file.
%% 
%% You are allowed to distribute this file under the condition that
%% it is distributed together with all files mentioned in readme.txt
%% and liesmich.txt. See LEGAL.TXT or LEGALDE.TXT for further
%% information.
%% 
%% If you receive only some of these files from someone, complain!
%% 
%% You are NOT ALLOWED to distribute this file alone.
%% 
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{scrartcl}
              [2001/08/17 v2.8l LaTeX2e KOMA
               document class]
\newcommand*\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\newif\if@hsl
\newif\if@fsl
\newif\if@abstrt
\newcommand\ifnumber[3]{
 \begingroup\@tempswafalse\let\next\test@number
 \expandafter\next#1\next
 \if@tempswa\endgroup#2\else\endgroup#3\fi
}
\def\test@number#1{
 \ifx \next#1
  \let\next\relax
 \else
  \@tempcnta=\expandafter\expandafter\expandafter\number\expandafter`#1\relax
  \ifnum \@tempcnta>47\relax
   \ifnum \@tempcnta<58\relax
    \@tempswatrue
   \else\@tempswafalse\fi
  \else\@tempswafalse\fi
  \if@tempswa\else\let\next\gobble@till@next\fi\fi
 \next}
\def\gobble@till@next#1\next{}
\def\@BCOR{\expandafter\@@BCOR\CurrentOption BCORBCOR\@@BCOR}
\def\@@BCOR #1BCOR#2BCOR#3\@@BCOR{
 \edef\@tempa{#1}\ifx\@tempa\@empty
  \edef\@tempa{#3}\def\@tempb{BCOR}\ifx\@tempa\@tempb
   \PassOptionsToPackage{BCOR#2}{typearea}
  \else\OptionNotUsed\fi
 \else\OptionNotUsed\fi}
\def\@DIV{\expandafter\@@DIV\CurrentOption DIVDIV\@@DIV}
\def\@@DIV #1DIV#2DIV#3\@@DIV{
 \edef\@tempa{#1}\ifx\@tempa\@empty
  \edef\@tempa{#3}\def\@tempb{DIV}\ifx\@tempa\@tempb
   \PassOptionsToPackage{DIV#2}{typearea}
  \else\OptionNotUsed\fi
 \else\@BCOR\fi}
\newcommand\@headlines{%
  \expandafter\@@headlines\CurrentOption headlines\@@headlines}
\def\@@headlines #1headlines#2\@@headlines{
  \edef\@tempa{#2}\edef\@tempb{headlines}\ifx\@tempa\@tempb
    \PassOptionsToPackage{#1headlines}{typearea}
  \else\@DIV\fi}
\def\@pt@scan{\expandafter\@pt@@scan\CurrentOption pt\@pt@@scan}
\def\@pt@@scan #1pt#2\@pt@@scan{
 \edef\@tempa{pt}\edef\@tempb{#2}\ifx\@tempa\@tempb
  \ifnumber{#1}
   {\@tempcnta=#1\relax
    \advance \@tempcnta by-10\relax
    \edef\@ptsize{\the\@tempcnta}}
   {\@headlines}
  \else\@headlines\fi}
\def\x@paper{\expandafter\x@@paper\CurrentOption paper\x@@paper}
\def\x@@paper #1#2paper#3\x@@paper{
 \edef\@tempa{paper}\edef\@tempb{#3}\ifx\@tempa\@tempb
  \ifx a#1\PassOptionsToPackage{a#2paper}{typearea}
  \else\ifx b#1\PassOptionsToPackage{b#2paper}{typearea}
  \else\ifx c#1\PassOptionsToPackage{c#2paper}{typearea}
  \else\ifx d#1\PassOptionsToPackage{d#2paper}{typearea}
  \else\OptionNotUsed\fi\fi\fi\fi
 \else\@pt@scan\fi}
\newif\if@tablecaptionabove\@tablecaptionabovefalse
\DeclareOption{tablecaptionabove}{\@tablecaptionabovetrue}
\DeclareOption{tablecaptionbelow}{\@tablecaptionabovefalse}
\newif\if@komalongtable\@komalongtabletrue
\DeclareOption{origlongtable}{\@komalongtablefalse}
\DeclareOption{headsepline}{\PassOptionsToPackage{headinclude}{typearea}
                            \@hsltrue}
\DeclareOption{headnosepline}{\@hslfalse}
\DeclareOption{footsepline}{\PassOptionsToPackage{footinclude}{typearea}
                            \@fsltrue}
\DeclareOption{footnosepline}{\@fslfalse}
\DeclareOption{10pt}{\renewcommand*\@ptsize{0}}
\DeclareOption{11pt}{\renewcommand*\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand*\@ptsize{2}}
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{notitlepage}{\@titlepagefalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand*\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand*\newblock{\par}}%
}
\DeclareOption{abstracton}{\@abstrttrue}
\DeclareOption{abstractoff}{\@abstrtfalse}
\newif\if@parskip\@parskipfalse
\newcommand*{\scr@parskip}{\setlength{\parskip}{\z@}}
\newcommand*{\scr@parfillskip}{\setlength{\parfillskip}{\z@}}
\DeclareOption{parskip}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .1\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{1em plus 1fil}}}
\DeclareOption{parskip-}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .1\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{0pt plus 1fil}}}
\DeclareOption{parskip+}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .1\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{.3333\linewidth}
    \addtolength{\parfillskip}{\z@ plus 1fil}}}
\DeclareOption{parskip*}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .1\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{.25\linewidth}
    \addtolength{\parfillskip}{\z@ plus 1fil}}}
\DeclareOption{halfparskip}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{.5\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .5\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{1em plus 1fil}}}
\DeclareOption{halfparskip-}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{.5\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .5\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{0pt plus 1fil}}}
\DeclareOption{halfparskip+}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{.5\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .5\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{.3333\linewidth}
    \addtolength{\parfillskip}{\z@ plus 1fil}}}
\DeclareOption{halfparskip*}{\@parskiptrue%
  \renewcommand*{\scr@parskip}{%
    \setlength{\parskip}{.5\baselineskip}%
    \addtolength{\parskip}{\z@ \@plus .5\baselineskip}}
  \renewcommand*{\scr@parfillskip}{%
    \setlength{\parfillskip}{.25\linewidth}
    \addtolength{\parfillskip}{\z@ plus 1fil}}}
\DeclareOption{parindent}{\@parskipfalse}
\DeclareOption{pointednumbers}
  {\AtEndOfClass{\@altsecnumformattrue
                 \global\let\@altsecnumformatfalse\@altsecnumformattrue}}
\DeclareOption{pointednumber}
  {\AtEndOfClass{\@altsecnumformattrue
                 \global\let\@altsecnumformatfalse\@altsecnumformattrue}
   \ClassError
    {scrartcl}
    {undefined option `pointednumber' instead of `pointednumbers'}
    {You've used the class-option `pointednumber'.\MessageBreak
     This name was an implementation-error at versions from 2.3h till
     2.4f.\MessageBreak
     The correct name is `pointednumbers'.
     Change this at your document.\MessageBreak
     If you'll continue, the option `pointednumbers' will be used.}}
\DeclareOption{pointlessnumbers}
  {\AtEndOfClass{\@altsecnumformatfalse
                 \global\let\@altsecnumformattrue\@altsecnumformatfalse}}
\DeclareOption{pointlessnumber}
  {\AtEndOfClass{\@altsecnumformatfalse
                 \global\let\@altsecnumformattrue\@altsecnumformatfalse}
   \ClassError
    {scrartcl}
    {undefined option `pointlessnumber' instead of `pointlessnumbers'}
    {You've used the class-option `pointlessnumber'.\MessageBreak
     This name was an implementation-error at versions from 2.3h till
     2.4f.\MessageBreak
     The correct name is `pointlessnumbers'.
     Change this at your document.\MessageBreak
     If you'll continue, the option `pointlessnumbers' will be used.}}
\newcommand\@bigheadings
  {\AtEndOfClass{
    \let\size@part\Huge
    \let\size@partnumber\huge
    \let\size@section\Large
    \let\size@subsection\large
    \let\size@subsubsection\normalsize
    \let\size@paragraph\normalsize
    \let\size@subparagraph\normalsize}}
\DeclareOption{bigheadings}{\@bigheadings}
\DeclareOption{bigheading}{\@bigheadings
  \ClassError
    {scrartcl}
    {undefined option `bigheading' instead of `bigheadings'}
    {You've used the old option name `bigheading'!\MessageBreak
     The correct name of the option is `bigheadings'.\MessageBreak
     The old name was an implementation error from\MessageBreak
     version 2.3h till 2.4f.\MessageBreak
     You should change this at your document!\MessageBreak
     If you'll continue, option `bigheadings' will be used.}}
\newcommand\@normalheadings
  {\AtEndOfClass{
    \let\size@part\huge
    \let\size@partnumber\huge
    \let\size@section\large
    \let\size@subsection\normalsize
    \let\size@subsubsection\normalsize
    \let\size@paragraph\normalsize
    \let\size@subparagraph\normalsize}}
\DeclareOption{normalheadings}{\@normalheadings}
\DeclareOption{normalheading}{\@normalheadings
  \ClassError
    {scrartcl}
    {undefined option `normalheading' instead of `normalheadings'}
    {You've used the old option name `normalheading'!\MessageBreak
     The correct name of the option is `normalheadings'.\MessageBreak
     The old name was an implementation error from\MessageBreak
     version 2.3h till 2.4f.\MessageBreak
     You should change this at your document!\MessageBreak
     If you'll continue, option `normalheadings' will be used.}}
\newcommand\@smallheadings
  {\AtEndOfClass{
    \let\size@part\LARGE
    \let\size@partnumber\LARGE
    \let\size@section\normalsize
    \let\size@subsection\normalsize
    \let\size@subsubsection\normalsize
    \let\size@paragraph\normalsize
    \let\size@subparagraph\normalsize}}
\DeclareOption{smallheadings}{\@smallheadings}
\DeclareOption{smallheading}{\@smallheadings
  \ClassError
    {scrartcl}
    {undefined option `smallheading' instead of `smallheadings'}
    {You've used the old option name `smallheading'!\MessageBreak
     The correct name of the option is `smallheadings'.\MessageBreak
     The old name was an implementation error from\MessageBreak
     version 2.3h till 2.4f.\MessageBreak
     You should change this at your document!\MessageBreak
     If you'll continue, option `smallheadings' will be used.}}
\newcommand*{\cleardoublestandardpage}{}
\let\cleardoublestandardpage\cleardoublepage
\newcommand*{\cleardoubleemptypage}{\clearpage%
  {\thispagestyle{empty}\cleardoublestandardpage}}
\newcommand*{\cleardoubleplainpage}{\clearpage%
  {\thispagestyle{plain}\cleardoublestandardpage}}
\DeclareOption{cleardoubleempty}{\let\cleardoublepage\cleardoubleemptypage}
\DeclareOption{cleardoubleplain}{\let\cleardoublepage\cleardoubleplainpage}
\DeclareOption{cleardoublestandard}{\let\cleardoublepage\cleardoublestandard}
\DeclareOption{liststotoc}{%
  \AtEndOfClass{%
    \renewcommand*{\float@listhead}[1]{%
      \addsec{#1}
      \@mkboth{#1}{#1}}}
}
\DeclareOption{bibtotoc}
  {\AtEndOfClass{%
    \renewcommand*\bib@heading{%
      \addsec{\refname}
      \@mkboth{\refname}{\refname}
  }}}
\DeclareOption{bibtotocnumbered}
  {\AtEndOfClass{%
    \renewcommand*\bib@heading{%
      \section{\refname}
      \@mkboth{\refname}{\refname}
  }}}
\DeclareOption{idxtotoc}
 {\AtEndOfClass{%
   \renewcommand*\idx@heading{%
     \if@twoside\cleardoublepage\else\clearpage\fi%
     \twocolumn[\addsec{\indexname}]
     \@mkboth{\indexname}{\indexname}%
 }}}
\if@compatibility\else
  \DeclareOption*{\x@paper}
\fi
\ExecuteOptions{11pt,onecolumn,final,headnosepline,footnosepline,bigheadings,%
oneside,abstractoff}
\ProcessOptions
\@tempcnta=10
\advance\@tempcnta by\@ptsize\relax
\xdef\@tempa{%
size%
\the\@tempcnta.clo}
\InputIfFileExists{\@tempa}{}{%
    \ClassError
    {scrartcl}
      {File `\@tempa' not found!\MessageBreak
       This error is fatal!}
      {To every fontsize you want to select by class option,\MessageBreak
       you need a corresponding
       size-class-option-file declaring all\MessageBreak
       the macros based on fontsize.\MessageBreak
       Ask a wizzard for more information or use another
       fontsize.\MessageBreak
       You cannot continue!}
       \batchmode \csname @@end\endcsname \end
}
\RequirePackage{typearea}[1995/06/25]
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\if@parskip
  \scr@parskip
  \setlength{\parindent}{\z@}
  \AtEndOfClass{\scr@parfillskip}
  \g@addto@macro{\@listi}{\topsep\z@ \parsep\parskip \itemsep\z@}
  \let\@listI\@listi
  \g@addto@macro{\@listii}{\topsep\z@ \parsep\parskip \itemsep\z@}
  \g@addto@macro{\@listiii}{\topsep\z@ \parsep\parskip \itemsep\z@}
  \g@addto@macro{\footnotesize}{%
    \def\@listi{\leftmargin\leftmargini
      \topsep \z@ \parsep \parskip \itemsep \z@}%
  }
  \g@addto@macro{\small}{%
    \def\@listi{\leftmargin\leftmargini
      \topsep \z@ \parsep \parskip \itemsep \z@}%
  }
\else
  \setlength{\parskip}{0\p@ \@plus \p@}
  \setlength{\parindent}{1em}
  \setlength{\parfillskip}{\z@ \@plus 1fil}
\fi
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand*\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand*\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand*\textfraction{.2}
\renewcommand*\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand*\dbltopfraction{.7}
\renewcommand*\dblfloatpagefraction{.5}
\if@compatibility
  \newcommand*\headincludeon{\@hincltrue}
  \newcommand*\headincludeoff{\@hinclfalse}
  \newcommand*\headseplineon{\@hsltrue \@hincltrue}
  \newcommand*\headseplineoff{\@hslfalse \@hinclfalse}
  \newcommand*\footincludeon{\@fincltrue}
  \newcommand*\footincludeoff{\@finclfalse}
  \newcommand*\footseplineon{\@fsltrue \@fincltrue}
  \newcommand*\footseplineoff{\@fslfalse \@finclfalse}
\fi
\if@twoside
 \def\ps@plain{\let\@mkboth\@gobbletwo
  \def\@evenhead{}%
  \def\@oddhead{}%
  \def\@evenfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\thepage\hfil}}}}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage}}}}}
 \def\ps@headings{\let\@mkboth\markboth
  \def\@evenhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\strut\leftmark\hfil}}%
   \if@hsl \vskip 1.5\p@ \hrule \fi}}
  \def\@oddhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\hfil\strut\rightmark}}%
   \if@hsl \vskip 1.5\p@ \hrule \fi}}
  \def\@evenfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\thepage\hfil}}}}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage}}}}%
  \def\sectionmark##1{%
   \markboth {\ifnum \c@secnumdepth >\z@%
    \sectionmarkformat\fi ##1}{}}
  \def\subsectionmark##1{%
   \markright {\ifnum \c@secnumdepth >\@ne%
    \subsectionmarkformat\fi ##1}}}
 \def\ps@myheadings{\let\@mkboth\@gobbletwo
  \def\@evenhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\strut\leftmark\hfil}}
   \if@hsl \vskip 1.5\p@ \hrule \fi}}%
  \def\@oddhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\hfil\strut\rightmark}}
   \if@hsl \vskip 1.5\p@ \hrule \fi}}%
  \def\@evenfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\thepage\hfil}}}}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage}}}}%
  \def\subsectionmark##1{}
  \def\sectionmark##1{}}
\else
 \def\ps@plain{\let\@mkboth\@gobbletwo
  \def\@evenhead{}%
  \def\@oddhead{}%
  \def\@evenfoot{}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage\hfil}}}}}
 \def\ps@headings{\let\@mkboth\markboth
  \def\@evenhead{}%
  \def\@oddhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\hfil\strut\rightmark\hfil}}
   \if@hsl \vskip 1.5\p@ \hrule \fi}}%
  \def\@evenfoot{}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage\hfil}}}}%
  \def\subsectionmark##1{}
  \def\sectionmark##1{\markright {\ifnum \c@secnumdepth >\z@
      \sectionmarkformat\fi ##1}}}
 \def\ps@myheadings{\let\@mkboth\@gobbletwo
  \def\@evenhead{}%
  \def\@oddhead{\vbox{\hsize=\textwidth
   \hb@xt@ \textwidth{{\headfont\hfil\strut\rightmark\hfil}}
   \if@hsl \vskip 1.5\p@ \hrule \fi}}%
  \def\@evenfoot{}%
  \def\@oddfoot{\vbox{\hsize=\textwidth
   \if@fsl \hrule \vskip 3\p@ \fi
   \hb@xt@ \textwidth{{\pnumfont\hfil\thepage\hfil}}}}%
  \def\subsectionmark##1{}%
  \def\sectionmark##1{}}
\fi
\newcommand*{\titlepagestyle}{plain}
\newcommand*{\partpagestyle}{plain}
\newcommand*{\indexpagestyle}{plain}
\newcommand*{\@extratitle}{}
\newcommand{\extratitle}[1]{\gdef\@extratitle{#1}}
\newcommand*{\@titlehead}{}
\newcommand{\titlehead}[1]{\gdef\@titlehead{#1}}
\newcommand*{\@subject}{}
\newcommand*{\subject}[1]{\gdef\@subject{#1}}
\newcommand*{\@publishers}{}
\newcommand{\publishers}[1]{\gdef\@publishers{#1}}
\newcommand*{\@uppertitleback}{}
\newcommand{\uppertitleback}[1]{\gdef\@uppertitleback{#1}}
\newcommand*{\@lowertitleback}{}
\newcommand{\lowertitleback}[1]{\gdef\@lowertitleback{#1}}
\newcommand*{\@dedication}{}
\newcommand{\dedication}[1]{\gdef\@dedication{#1}}
\newcommand*{\next@tpage}{\clearpage\thispagestyle{empty}}
\if@titlepage
    \newcommand*\maketitle[1][1]{\begin{titlepage}%
    \setcounter{page}{#1}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let\footnote\thanks
    \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
    \let\@oldmakefnmark\@makefnmark
    \def\@makefnmark{\rlap\@oldmakefnmark}
    \ifx\@extratitle\@empty \else
        \noindent\@extratitle\next@tpage\cleardoublepage\thispagestyle{empty}
    \fi
    \ifx\@titlehead\@empty \else
        \noindent\begin{minipage}[t]{\textwidth}
        \@titlehead
        \end{minipage}\par
    \fi
    \null\vfill
    \begin{center}
    \ifx\@subject\@empty \else
        {\Large \@subject \par}
        \vskip 3em
    \fi
    {\sectfont\huge \@title\par}
    \vskip 3em
    {\Large \lineskip 0.75em
    \begin{tabular}[t]{c}
        \@author
    \end{tabular}\par}
    \vskip 1.5em
    {\Large \@date \par}
    \vskip \z@ \@plus3fill
    {\Large \@publishers \par}
    \vskip 3em
    \end{center}\par
    \@thanks
    \vfill\null
    \if@twoside\next@tpage
        \noindent\begin{minipage}[t]{\textwidth}
        \@uppertitleback
        \end{minipage}\par
        \vfill
        \noindent\begin{minipage}[b]{\textwidth}
        \@lowertitleback
        \end{minipage}
    \fi
    \ifx\@dedication\@empty \else
        \next@tpage\null\vfill
        {\centering \Large \@dedication \par}
        \vskip \z@ \@plus3fill
        \if@twoside \next@tpage\cleardoublepage \fi
    \fi
    \end{titlepage}
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\@extratitle\@empty
    \global\let\@titlehead\@empty
    \global\let\@subject\@empty
    \global\let\@publishers\@empty
    \global\let\@uppertitleback\@empty
    \global\let\@lowertitleback\@empty
    \global\let\@dedication\@empty
    \global\let\author\relax
    \global\let\title\relax
    \global\let\extratitle\relax
    \global\let\titlehead\relax
    \global\let\subject\relax
    \global\let\publishers\relax
    \global\let\uppertitleback\relax
    \global\let\lowertitleback\relax
    \global\let\dedication\relax
    \global\let\date\relax
    \global\let\and\relax}
\else
     \newcommand*\maketitle[1][1]{\par
       \@tempcnta=#1\relax\ifnum\@tempcnta=1\else
         \ClassWarning{scrartcl}
                      {Optional argument of \string\maketitle\space ignored
                       at\MessageBreak
                       notitlepage-mode}\fi
       \begingroup
         \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
         \let\@oldmakefnmark\@makefnmark
         \def\@makefnmark{\rlap\@oldmakefnmark}
         \if@twocolumn
           \ifnum \col@number=\@ne
             \@maketitle
           \else
             \twocolumn[\@maketitle]%
           \fi
         \else
           \newpage
           \global\@topnum\z@   % Prevents figures from going at top of page.
           \@maketitle
         \fi
         \thispagestyle{\titlepagestyle}\@thanks
       \endgroup
       \setcounter{footnote}{0}%
       \let\thanks\relax
       \let\maketitle\relax
       \let\@maketitle\relax
       \global\let\@thanks\@empty
       \global\let\@author\@empty
       \global\let\@date\@empty
       \global\let\@title\@empty
       \global\let\@extratitle\@empty
       \global\let\@titlehead\@empty
       \global\let\@subject\@empty
       \global\let\@publishers\@empty
       \global\let\@uppertitleback\@empty
       \global\let\@lowertitleback\@empty
       \global\let\@dedication\@empty
       \global\let\author\relax
       \global\let\title\relax
       \global\let\extratitle\relax
       \global\let\titlehead\relax
       \global\let\subject\relax
       \global\let\publishers\relax
       \global\let\uppertitleback\relax
       \global\let\lowertitleback\relax
       \global\let\dedication\relax
       \global\let\date\relax
       \global\let\and\relax}
\fi
\def\@maketitle{%
    \clearpage
    \let\footnote\thanks
    \ifx\@extratitle\@empty \else
        \noindent\@extratitle \next@tpage \if@twoside \null\next@tpage \fi
    \fi
    \ifx\@titlehead\@empty \else
        \noindent\begin{minipage}[t]{\textwidth}
        \@titlehead
        \end{minipage}\par
    \fi
    \null
    \vskip 2em%
    \begin{center}%
    \ifx\@subject\@empty \else
        {\Large \@subject \par}
        \vskip 1.5em
    \fi
    {\sectfont\huge \@title \par}%
    \vskip 1.5em%
    {\Large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\Large \@date \par}%
    \vskip \z@ \@plus 1em
    {\Large \@publishers \par}
    \ifx\@dedication\@empty \else
        \vskip 2em
        {\Large \@dedication \par}
    \fi
  \end{center}%
  \par
  \vskip 2em}
\renewcommand*{\numberline}[1]{%
  \hb@xt@\@tempdima{#1\autodot\hfil}}
\setcounter{secnumdepth}{3}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand*\thepart          {\@Roman\c@part}
\renewcommand*\thesection       {\@arabic\c@section}
\renewcommand*\thesubsection    {\thesection.\@arabic\c@subsection}
\renewcommand*\thesubsubsection {\thesubsection.\@arabic\c@subsubsection}
\renewcommand*\theparagraph     {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand*\thesubparagraph  {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{\par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \@maybeautodot\thepart%
      \addcontentsline{toc}{part}{\protect\numberline{\thepart}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \size@partnumber\sectfont\partformat
       \par\nobreak
     \fi
     \size@part\sectfont #2\strut%
     \@mkboth{}{}\par}%
     \nobreak
     \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \size@part\sectfont #1\strut\@mkboth{}{}\par}%
     \nobreak
     \vskip 3ex
    \@afterheading}
\newcommand\addsec{\secdef\@addsec\@saddsec}
\def\@addsec[#1]#2{\section*{#2}\addcontentsline{toc}{section}{#1}
 \if@twoside \@mkboth{#1}{}\else \@mkboth{#1}{#1}\fi}
\def\@saddsec#1{\section*{#1}\@mkboth{}{}}
\newcommand\section{\@startsection{section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\raggedsection\normalfont\size@section\sectfont}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\raggedsection\normalfont\size@subsection\sectfont}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\raggedsection\normalfont\size@subsubsection\sectfont}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
  {3.25ex \@plus1ex \@minus.2ex}%
  {-1em}%
  {\raggedsection\normalfont\size@paragraph\sectfont}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
  {3.25ex \@plus1ex \@minus .2ex}%
  {-1em}%
  {\raggedsection\normalfont\size@subparagraph\sectfont}}
\newcommand\minisec[1]{\@afterindentfalse \vskip 1.5ex
  {\parindent \z@ \raggedsection\sectfont #1\par\nobreak}%
  \@afterheading}
\let\raggedsection\raggedright
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength\leftmargin    {\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand*\theenumi{\@arabic\c@enumi}
\renewcommand*\theenumii{\@alph\c@enumii}
\renewcommand*\theenumiii{\@roman\c@enumiii}
\renewcommand*\theenumiv{\@Alph\c@enumiv}
\newcommand*\labelenumi{\theenumi.}
\newcommand*\labelenumii{\theenumii)}
\newcommand*\labelenumiii{\theenumiii.}
\newcommand*\labelenumiv{\theenumiv.}
\renewcommand*\p@enumii{\theenumi}
\renewcommand*\p@enumiii{\p@enumii(\theenumii)}
\renewcommand*\p@enumiv{\p@enumiii\theenumiii}
\newcommand*\labelitemi{\textbullet}
\newcommand*\labelitemii{\normalfont\bfseries\textendash}
\newcommand*\labelitemiii{\textasteriskcentered}
\newcommand*\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*{\descriptionlabel}[1]{\hspace\labelsep
                                \descfont #1}
\newenvironment{labeling}[2][]
  {\def\sc@septext{#1}
   \list{}{\settowidth{\labelwidth}{#2#1}
     \leftmargin\labelwidth \advance\leftmargin by \labelsep
     \let\makelabel\labelinglabel}}
  {\endlist}
\newcommand\labelinglabel[1]{#1\hfil\sc@septext}
\newenvironment{verse}
               {\let\\=\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@titlepage
 \newenvironment{abstract}{%
     \titlepage
     \null\vfil
     \@beginparpenalty\@lowpenalty
     \if@abstrt
     \begin{center}
       \sectfont \abstractname
       \@endparpenalty\@M
     \end{center}\fi}
    {\par\vfil\null\endtitlepage}
\else
 \newenvironment{abstract}{%
     \if@twocolumn\if@abstrt
       \addsec*{\abstractname}\fi%
     \else
       \if@abstrt
       \small
       \begin{center}%
         {\sectfont \abstractname\vspace{-.5em}\vspace{\z@}}%
       \end{center}\fi%
       \quotation
     \fi}
     {\if@twocolumn\else\endquotation\fi}
\fi
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \if@compatibility
        \setcounter{page}{0}
      \fi}%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\newcommand\marginline[1]{\marginpar[\raggedleft #1]{\raggedright #1}}
\newcommand*\appendix{%
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \renewcommand*\thesection{\@Alph\c@section}%
  \@ifundefined{appendixmore}{}{\appendixmore}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand*\theequation{%
                          \@arabic\c@equation}
\newcounter{figure}
\renewcommand*\thefigure{%
                        \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figureformat}
\newenvironment{figure}
               {\let\caption\captionbelow\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\let\caption\captionbelow\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand*\thetable{%
                       \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tableformat}
\newenvironment{table}
               {\if@tablecaptionabove\let\caption\captionabove%
                 \else\let\caption\captionbelow\fi\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\if@tablecaptionabove\let\caption\captionabove%
                 \else\let\caption\captionbelow\fi\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\newif\if@captionabove\@captionabovefalse
\newcommand{\scr@caption}{}
\AtBeginDocument{\let\scr@caption\caption}
\newcommand{\captionabove}{\@captionabovetrue\scr@caption}
\newcommand{\captionbelow}{\@captionabovefalse\scr@caption}
\newif\ifdofullc@p
\dofullc@ptrue
\newif\if@capbreak\@capbreakfalse
\newcommand*{\setcapindent}[1]{}
\def\setcapindent{\@ifstar{\@@setcapindent}{\@setcapindent}}
\newcommand*{\@setcapindent}[1]{
  \setlength{\@tempdima}{#1}%
  \ifdim\@tempdima>\hsize\addtolength{\@tempdima}{-\hsize}\@capbreaktrue
  \else\@capbreakfalse\fi
  \edef\cap@indent{\the\@tempdima}
  \dofullc@pfalse
}
\newcommand*{\@@setcapindent}[1]{
  \setlength{\@tempdima}{#1}%
  \edef\cap@indent{\the\@tempdima}
  \dofullc@pfalse\@capbreaktrue
}
\newcommand*{\setcaphanging}{\dofullc@ptrue\@capbreakfalse}
\newcommand{\@makecaption}[2]{%
  \if@captionabove\vskip\belowcaptionskip\else\vskip\abovecaptionskip\fi
  \@@makecaption\@firstofone{#1}{#2}
  \if@captionabove\vskip\abovecaptionskip\else\vskip\belowcaptionskip\fi}
\newcommand{\@@makecaption}[3]{%
  \sbox\@tempboxa{{\capfont#1{{\caplabelfont #2\captionformat}}\capfont #3}}%
  \ifdim \wd\@tempboxa >\hsize
    \ifdofullc@p
      {\capfont\@hangfrom{#1{{\caplabelfont #2\captionformat}}}{#3\par}}
    \else%
      \ifdim\cap@indent<\z@%
        {\capfont#1{{\caplabelfont #2\captionformat\par}}\noindent%
          \hspace*{-\cap@indent}#3\par}
      \else%
        \if@capbreak
          {\capfont#1{{\caplabelfont #2\captionformat\par}}\noindent%
            \@hangfrom{\hspace*{\cap@indent}}{#3\par}}
        \else
          {\capfont\@hangfrom{\hspace*{\cap@indent}}%
            {\hspace{-\cap@indent}#1{{\caplabelfont #2\captionformat}}#3\par}}
        \fi
      \fi
    \fi
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi}
\AtBeginDocument{%
  \ifx\LT@makecaption\undefined\else
    \@ifpackageloaded{caption2}{\@komalongtablefalse%
      \ClassInfo{%
       scrartcl}{%
        no change of longtable captions because\MessageBreak
        package ``caption2'' detected}}{}
    \if@komalongtable\ClassInfo{%
        scrartcl}{%
        longtable captions redefined}
      \renewcommand{\LT@makecaption}[3]{%
        \LT@mcol\LT@cols c{%
          \edef\outerlinewidth{\the\linewidth}%
          \hbox to\z@{\hss\parbox[t]\LTcapwidth{%
              \@@makecaption{#1}{#2}{#3}
              \endgraf\vskip\baselineskip}%
            \hss}}}
    \fi
  \fi
}
\newcommand*{\fs@komaabove}{%
  \let\@fs@capt\floatc@komascript
  \let\@fs@pre\@empty
  \def\@fs@mid{\vskip\abovecaptionskip}
  \let\@fs@post\@empty
  \let\@fs@iftopcapt\iftrue}
\newcommand*{\fs@komabelow}{%
  \let\@fs@capt\floatc@komascript
  \let\@fs@pre\@empty
  \def\@fs@mid{\vskip\abovecaptionskip}
  \let\@fs@post\@empty
  \let\@fs@iftopcapt\iffalse}
\newcommand{\floatc@komascript}{\@@makecaption\@firstofone}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareOldFontCommand{\sfb}{\normalfont\sffamily\bfseries}{\@nomath\sfb}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand*\descfont{\sffamily\bfseries}
\newcommand*\sectfont{\normalfont\normalcolor\sffamily\bfseries}
\newcommand*\pnumfont{\normalfont\normalcolor}
\newcommand*\headfont{\normalfont\normalcolor\slshape}
\newcommand*\capfont{\normalfont\normalcolor}
\newcommand*\caplabelfont{\normalfont\normalcolor}
\newcommand*\@pnumwidth{1.55em}
\newcommand*\@tocrmarg {2.55em}
\newcommand*\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand*\toc@heading{\section*{\contentsname
  \@mkboth{\contentsname}{\contentsname}}}
\newcommand*\tableofcontents{%
  \begingroup
    \toc@heading%
    \setlength{\parskip}{\z@}%
    \@starttoc{toc}%
  \endgroup
}
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{\@secpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{2em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode\sectfont\large
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \if@compatibility
        \global\@nobreaktrue
        \everypar{\global\@nobreakfalse\everypar{}}%
      \else
      \penalty\@highpenalty
       \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty{\@secpenalty}%
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \sectfont
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand*{\float@listhead}[1]{%
  \section*{#1}%
  \@mkboth{#1}{#1}%
}
\newcommand*\lof@heading{\float@listhead{\listfigurename}}
\newcommand*\listoffigures{%
  \begingroup%
    \lof@heading%
    \setlength{\parskip}{\z@}%
    \@starttoc{lof}%
  \endgroup
}
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\lot@heading{\float@listhead{\listtablename}}
\newcommand*\listoftables{%
  \begingroup
    \lot@heading%
    \setlength{\parskip}{\z@}%
    \@starttoc{lot}%
  \endgroup
}
\let\l@table\l@figure
\newcommand*{\bibpreamble}{\relax}
\newcommand{\setbibpreamble}[1]{\def\bibpreamble{#1}}
\newcommand*\bib@heading{%
  \section*{\refname}
  \@mkboth{\refname}{\refname}
}
\newdimen\bibindent
\bibindent=1.5em
\newenvironment{thebibliography}[1]{%
  \bib@heading%
  \noindent\bibpreamble\par\nobreak
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand*\theenumiv{\@arabic\c@enumiv}}%
  \sloppy\clubpenalty4000\widowpenalty4000%
  \sfcode`\.=\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
\newcommand*\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newcommand*{\index@preamble}{\relax}
\newcommand{\setindexpreamble}[1]{\renewcommand*{\index@preamble}{#1}}
\newcommand*{\idx@heading}{%
  \if@twoside\cleardoublepage\else\clearpage\fi%
  \twocolumn[\section*{\indexname}]
  \@mkboth{\indexname}{\indexname}%
}
\newenvironment{theindex}
  {\if@twocolumn
    \@restonecolfalse
    \else
    \@restonecoltrue
    \fi
    \columnseprule \z@
    \columnsep 35\p@
    \idx@heading%
    \index@preamble\par\nobreak
    \thispagestyle{\indexpagestyle}\parindent\z@
    \parskip\z@ \@plus .3\p@\relax
    \let\item\@idxitem}
  {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand*\@idxitem  {\par\hangindent 40\p@}
\newcommand*\subitem   {\@idxitem \hspace*{20\p@}}
\newcommand*\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand*\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand*\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern 2.6\p@}
\newcommand\deffootnote[4][]{%
  \long\def\@makefntext##1{%
    \edef\@tempa{#1}\ifx\@tempa\@empty
      \@setpar{\@@par
        \@tempdima = \hsize
        \addtolength{\@tempdima}{-#2}
        \parshape \@ne #2 \@tempdima}%
    \else
      \@setpar{\@@par
        \@tempdima = \hsize
        \addtolength{\@tempdima}{-#1}
        \@tempdimb = \hsize
        \addtolength{\@tempdimb}{-#2}
        \parshape \tw@ #1 \@tempdima #2 \@tempdimb}%
    \fi
    \par
    \parindent #3\noindent
    \hbox to \z@{\hss\@@makefnmark}##1}
  \def\@@makefnmark{\hbox{#4}}
}
\newcommand*\deffootnotemark[1]{\def\@makefnmark{\hbox{#1}}}
\def\thefootnotemark{\@thefnmark}
\deffootnote[1em]{1.5em}{1em}
  {\textsuperscript{\normalfont\normalcolor\thefootnotemark}}
\deffootnotemark{\textsuperscript{\normalfont\normalcolor\thefootnotemark}}
\DeclareRobustCommand*\textsubscript[1]{%
  \@textsubscript{\selectfont#1}}
\newcommand{\@textsubscript}[1]{%
  {\m@th\ensuremath{_{\mbox{\fontsize\sf@size\z@#1}}}}}
\newcommand*\contentsname{Contents}
\newcommand*\listfigurename{List of Figures}
\newcommand*\listtablename{List of Tables}
\newcommand*\refname{References}
\newcommand*\indexname{Index}
\newcommand*\figurename{Figure}
\newcommand*\tablename{Table}
\newcommand*\partname{Part}
\newcommand*\appendixname{Appendix}
\newcommand*\abstractname{Abstract}
\newcommand*\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\newif\if@altsecnumformat\@altsecnumformatfalse
\newif\if@autodot\@autodotfalse
\newcommand{\@maybeautodot}[1]{\if@autodot\else\begingroup%
  \expandafter\@@maybeautodot #1\@stop\endgroup\fi}
\def\@@maybeautodot#1{%
  \ifx #1\@stop\let\@@maybeautodot\relax%
  \else%
    \ifx #1\Alph \aftergroup\@autodottrue\fi%
    \ifx #1\alph \aftergroup\@autodottrue\fi%
    \ifx #1\Roman \aftergroup\@autodottrue\fi%
    \ifx #1\roman \aftergroup\@autodottrue\fi%
    \ifx #1\@Alph \aftergroup\@autodottrue\fi%
    \ifx #1\@alph \aftergroup\@autodottrue\fi%
    \ifx #1\@Roman \aftergroup\@autodottrue\fi%
    \ifx #1\@roman \aftergroup\@autodottrue\fi%
    \ifx #1\romannumeral \aftergroup\@autodottrue\fi%
  \fi%
  \@@maybeautodot}
\let\scr@sect\@sect
\def\@sect#1{\stepcounter{#1}%
  \expandafter\@maybeautodot\csname the#1\endcsname%
  \addtocounter{#1}{-1}\scr@sect{#1}}
\AtEndDocument{%
  \if@autodot\if@filesw\write\@auxout{%
    \string\global\string\@altsecnumformattrue}
  \fi\fi}
\newcommand*\autodot{\if@altsecnumformat.\fi}
\newcommand*\sectionmarkformat    {\thesection\autodot\enskip}
\newcommand*\subsectionmarkformat {\thesubsection\autodot\enskip}
\newcommand*\partformat    {\partname~\thepart\autodot}
\newcommand*\figureformat  {\figurename~\thefigure%
  }
\newcommand*\tableformat   {\tablename~\thetable%
}
\newcommand*\captionformat {:\ }
\newcommand*{\othersectionlevelsformat}[1]{%
  \csname the#1\endcsname\autodot\enskip}
\def\@seccntformat{\othersectionlevelsformat}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}      % Arabic page numbers
\if@twoside
\else
     \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
\@ifundefined{KOMAScipt}{%
  \DeclareRobustCommand{\KOMAScript}{\textsf{K\kern.05em O\kern.05em%
      M\kern.05em A\kern.1em-\kern.1em Script}}}{}
\endinput
%%
%% End of file `scrartcl.cls'.
