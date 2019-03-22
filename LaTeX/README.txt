
Title:   A Tractable Model of Buffer Stock Saving

Authors: Christopher D. Carroll and Patrick Toche

The file ctDiscrete.tex is a LaTeX file which should be compiled using  pdfLaTeX.  The entire directory structure of which the file is a part  is necessary for proper compilation.

### About LaTeX Distributions and Operating Systems

## This project was compiled with the 2012 TeXLive distribution. Also tested with TeXLive 2010, 2011, and 2012 on MacOSX, Ubuntu12, Windows7. Also tested with MikTeX29 on Windows.

## Compilation of the LaTeX document requires LaTeX to have permissions to write to files outside of its own directory. 

To allow write permissions in texlive, add the following lines to your texmf.cnf file:

openout_any = a
shell_escape = t

# On a standard MacTeX distribution, this file is located at:

/usr/local/texlive/[year]/texmf.cnf

# On a standard Windows installation, this file is located at:

C:\texlive\[year]\texmf.cnf

# On a standard Linux installation, this file may be located at:

/usr/local/texlive
/usr/share/texlive
/usr/bin/texlive

(The installation folder is sometimes named texmf-texlive)

where [year] is, e.g., 2010 if you have TeXLive-2010 installed.

Depending on details of your installation, you may need to modify other security preferences as well.

## If obtaining permissions is a problem on your system, a workaround is to move the Tables folder into the LaTeX folder and change every instance of:

\include{../Tables/etc}%
to:
\include{./Tables/etc}%


### About the siunitx package

## Our code requires version 2 of the siunitx package (and won't work with version 1).

## The current texlive distribution on debian and ubuntu still ships version 1 of siunitx, whereas our code uses version 2. To have an up-to-date distribution of texlive, install "vanilla" texlive. Then you can install packages using the tlmgr package manager. After successfully installing texlive, here are a few common operations with tlmgr:

$ tlmgr option repository http://mirror.ctan.org/systems/texlive/tlnet
$ tlgmr install <package name>
$ tlgmr update <package name>

Help:
http://www.tug.org/texlive/doc/tlmgr.html

##  The siunitx package is needed to typeset the tables, thus if the tables are commented out, the package is no longer needed.