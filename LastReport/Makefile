TARGET=j15416
EPS=$(shell ls *.eps)
TEX=platex -kanji=utf8
DVIPDF=dvipdfmx -p a4

.SUFFIXES: .tex .dvi

all:$(TARGET).dvi

$(TARGET).dvi: $(TARGET).tex $(EPS)

.tex.dvi:
	$(TEX) $<
	$(TEX) $<
	$(TEX) $<
.dvi.pdf:
	 dvipdfm $(PDFOPT) $<

tar:
	tar cvzf $(TARGET).tar.gz $(TARGET).* *.eps

pdf: $(TARGET).dvi
	$(DVIPDF) $(TARGET)

clean:
	rm -f $(TARGET).dvi $(TARGET).aux $(TARGET).log $(TARGET).toc
	rm -f $(TARGET).pdf $(TARGET).tar.gz
	rm -f *.pbm *.bmc
