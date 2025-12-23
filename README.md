**Quick Links**:
- [Interaktive Webseite](https://mib1213.github.io/MachineLearningLab/)
- [Notebook](https://github.com/mib1213/MachineLearningLab/blob/main/notebook.ipynb)
- [Arbeit PDF](https://github.com/mib1213/MachineLearningLab/blob/main/main.pdf)


# Inhaltsverzeichnis
* [0. Hintergrund](#hintergrund)
  * [Langschwanzpinguine (Pygoscelis)](#langschwanzpinguine-pygoscelis)
  * [Schnabelmessung](#schnabelmessung)

* [1. Einleitung](#1-einleitung)
  * [Daten](#daten)
    * [Überblick df.head](#überblick-dfhead)
  * [Featurebeschreibung](#featurebeschreibung)
    * [Zielvariable](#zielvariable)
    * [Numerische Merkmale](#numerische-merkmale)
    * [Kategoriale Merkmale](#kategoriale-merkmale)
  * [Train Test Split](#train-test-split)

* [2. EDA](#2-eda)
  * [Farbcode](#farbcode)
  * [Klassenverteilung](#klassenverteilung)
  * [Fehlende Werte](#fehlende-werte)
  * [Visualisierungen](#visualisierungen)
    * [Numerische Features](#numerische-features)
    * [Kategorische Features](#kategorische-features)
    * [Numerisch x Kategorisch](#numerisch-x-kategorisch)
  * [Korrelationsmatrix](#korrelationsmatrix)
  * [Feature Engineering](#feature-engineering)
  * [Outliers](#outliers)

* [3. Modellierung](#3-modellierung)
  * [GNB](#gnb)
    * [LOOCV](#loocv)
    * [CV](#cv)
    * [Entscheidungsgrenze](#entscheidungsgrenze)
  * [kNN](#knn)
    * [Auswahl von k](#auswahl-von-k)
    * [LOOCV Feature Permutation](#loocv-feature-permutation-knn)
    * [SHAP](#shap)
    * [CV](#cv-1)
    * [Entscheidungsgrenze](#entscheidungsgrenze-1)
  * [Entscheidungsbaum](#entscheidungsbaum)
    * [LOOCV Feature Permutation](#loocv-feature-permutation-dtc)
    * [Visualisierung](#visualisierung)
    * [Prunen](#prunen)
    * [Manueller Baum](#manueller-baum)
      * [Visualierung](#visualierung)
      * [LOOCV](#loocv-1)
      * [CV](#cv-2)
      * [Entscheidungsgrenze](#entscheidungsgrenze-2)

* [4. Modellauswahl](#4-modellauswahl)
  * [Entscheidungsgrenzen](#entscheidungsgrenzen)
  * [Konfusionsmatrizen](#konfusionsmatrizen)
  * [Benchmarking](#benchmarking)
  * [Bootstrapping](#bootstrapping)

* [5. Auswertung](#5-auswertung)
  * [Evaluation](#evaluation)
  * [Interpretation](#interpretation)

* [6. Fazit](#6-fazit)

* [7. Reflexion](#7-reflexion)

# Hintergrund
## Langschwanzpinguine (Pygoscelis)

Die Langschwanzpinguine sind eine Vogelgattung innerhalb der Familie der Pinguine. Das „Langschwanz“ in ihrem Namen bezieht sich auf ihre langen, steifen Schwanzfedern. Innerhalb der Langschwanzpinguine unterscheidet man drei lebende Arten: den Eselspinguin (Gentoo), seltener auch Rotschnabelpinguin genannt, den Adeliepinguin (Adelie) und den Zügelpinguin (Chinstrap), der auch Kehlstreifpinguin genannt wird. Da diese drei Arten alle an der Antarktischen Halbinsel brüten, werden sie auch als „Antarktisches Trio“ bezeichnet. [Wikipedia](https://de.wikipedia.org/wiki/Langschwanzpinguine)

<table align="center">
  <tr>
    <th id="adelie" align="center">Adelie</th>
    <th id="chinstrap" align="center">Chinstrap</th>
    <th id="gentoo" align="center">Gentoo</th>
  </tr>
  <tr>
    <td align="center">
      <img src="figs/Adelie.png" style="height:220px; width:auto; object-fit:contain;" />
    </td>
    <td align="center">
      <img src="figs/Chinstrap.png" style="height:220px; width:auto; object-fit:contain;" />
    </td>
    <td align="center">
      <img src="figs/Gentoo.png" style="height:220px; width:auto; object-fit:contain;" />
    </td>
  </tr>
  <tr>
    <td valign="top">
      Der Adelie Pinguin ist eher klein und kompakt gebaut. Er hat einen schwarzen Kopf mit einem deutlich sichtbaren weißen Ring um die Augen. Sein Schnabel ist kurz und kräftig.
    </td>
    <td valign="top">
      Der Chinstrap Pinguin ist ungefähr so groß wie der Adelie Pinguin, lässt sich aber leicht an einem schwarzen Streifen unter dem Kinn erkennen. Dieser Streifen sieht aus wie ein Kinnriemen (Chinstrap), daher auch der Name.
    </td>
    <td valign="top">
      Der Gentoo Pinguin ist der größte der drei Arten. Er hat einen langen Schnabel, der orange bis rötlich orange gefärbt ist. Auffällig sind auch seine sehr leuchtenden, orange roten Füße.
    </td>
  </tr>
</table>


<a id="schnabelmessung"></a>
<h2 align="center">Schnabelmessung</h2>
<p align="center">
  <img src="figs/culmen_depth.png" width="60%" />
</p>
<p align="center"><em>Artwork by @allison_horst</em></p>


# 1. Einleitung

Unser Ziel ist es, die Pinguinart vorherzusagen. Dabei liegt der Schwerpunkt ganz klar auf einer transparenten und gut nachvollziehbaren Modellierung. Interpretierbarkeit hat Vorrang vor reiner Vorhersageleistung. Machine Learning setzen wir nur dann ein, wenn klassische, leicht nachvollziehbare statistische Verfahren an ihre Grenzen stoßen.

## Daten

Der Datensatz [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) basiert auf realen Feldmessungen, die ursprünglich von [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) im Rahmen des [Palmer Station Long Term Ecological Research (LTER) Programms](https://pallter.marine.rutgers.edu/) in der Antarktis erhoben und anschließend öffentlich zur Verfügung gestellt wurden.

### Überblick `df.head`

| species   | island    |   bill_length_mm |   bill_depth_mm |   flipper_length_mm |   body_mass_g | sex    |
|-----------|-----------|------------------|-----------------|---------------------|---------------|--------|
| Adelie    | Torgersen |             39.1 |            18.7 |                 181 |          3750 | Male   |
| Adelie    | Torgersen |             39.5 |            17.4 |                 186 |          3800 | Female |
| Adelie    | Torgersen |             40.3 |            18   |                 195 |          3250 | Female |
| Adelie    | Torgersen |            nan   |           nan   |                 nan |           nan | nan    |
| Adelie    | Torgersen |             36.7 |            19.3 |                 193 |          3450 | Female |

## Featurebeschreibung

Der Palmer Penguins Datensatz enthält morphologische Messungen von 344 Pinguinen aus der Antarktis.

### Zielvariable
**species**: Kategoriale Zielvariable mit drei Klassen (Werte: Adelie, Chinstrap, Gentoo)

### Numerische Merkmale

**bill_length_mm**: Länge des Schnabels in Millimetern

**bill_depth_mm**: Tiefe (Höhe) des Schnabels in Millimetern

**flipper_length_mm**: Länge der Flosse in Millimetern

**body_mass_g**: Körpergewicht in Gramm

### Kategoriale Merkmale

**sex**: Geschlecht des Pinguins (Werte: male, female)

**island**: Insel, auf der der Pinguin beobachtet wurde (Werte: Biscoe, Dream, Torgersen)

## Train Test Split

Ganz am Anfang trennen wir das Testset (20%) stratifiziert vollständig ab, um jegliche Form von Data Leakage zu vermeiden. Im gesamten Notebook arbeiten wir ausschließlich mit den Trainingsdaten. Das Testset wird erst ganz am Ende verwendet, um die finale Modellperformance zu evaluieren.

# 2. EDA

## Farbcode

In der gesamten Arbeit werden die Spezies konsistent wie folgt eingefärbt:

| Spezies | Farbe |
|---|---|
| **Adelie** | ![](https://img.shields.io/badge/Adelie-%20-008fd5?style=flat-square&labelColor=008fd5) |
| **Chinstrap** | ![](https://img.shields.io/badge/Chinstrap-%20-e5ae38?style=flat-square&labelColor=e5ae38) |
| **Gentoo** | ![](https://img.shields.io/badge/Gentoo-%20-fc4f30?style=flat-square&labelColor=fc4f30) |

## Klassenverteilung

![Klassenverteilung](figs/image-5.png)

Die Klassen sind leicht ungleich verteilt. Grundsätzlich könnte man hier Oversampling Verfahren wie `RandomOverSampler` oder `SMOTE` aus dem Package `imbalanced-learn` einsetzen, um dieses Ungleichgewicht auszugleichen. Solche Methoden bringen jedoch eigene Herausforderungen mit sich. Deshalb erscheint es sinnvoll, zunächst ohne Oversampling weiterzuarbeiten und diese Verfahren erst dann einzusetzen, wenn sich das Klassenungleichgewicht tatsächlich als relevantes Problem in der Modellperformance zeigt.

## Fehlende Werte

![Fehlende Werte](figs/image-1.png)

Wir sehen, dass für alle vier numerischen Messungen jeweils 2 Werte fehlen. Eine wichtige Frage ist dabei, ob diese fehlenden Werte zu denselben Datenpunkten gehören. Zusätzlich fehlen 11 Werte für das Feature `sex`.

![MSNO Matrix](figs/image-2.png)

Man sieht in der MSNO Matrix, dass bei denselben zwei Datenpunkten nicht nur alle vier numerischen Messungen fehlen, sondern auch das Geschlecht. Daher würde ich diese Datenpunkte einfach droppen.

In der Produktion und bei den Testdaten könnten ebenfalls fehlende Werte auftreten, das wissen wir zum jetzigen Zeitpunkt noch nicht. Solche Fälle ließen sich über eine Fallback Logik abfangen. Fehlen alle Hauptfeatures, könnte das Modell direkt die Majoritätsklasse zurückgeben, in unserem Fall `Adelie`, was immerhin in 44.15% der Fälle korrekt wäre. Dabei ist wichtig, diese Logik transparent gegenüber den Stakeholdern zu kommunizieren, damit klar ist, wie Entscheidungen im Hintergrund zustande kommen.

Alternativ könnte man in solchen Situationen bewusst eine Fehlermeldung ausgeben. Beide Ansätze vermeiden, dass durch Imputation potenziell unrealistische Werte ins Modell gelangen. In diesem Kontext ist es besser, für einzelne Datenpunkte keine Vorhersage zu liefern, als dem Modell künstliche Muster beizubringen, die in der Realität nicht existieren.

Hinweis:
Alternativ könnte man versuchen, anhand der Insel die bedingte Wahrscheinlichkeit für die Art zu schätzen oder im Preprocessing einen *KNNImputer* einsetzen, um fehlende Werte durch einigermaßen realistische Werte zu ersetzen. Da es sich aber nur um zwei Datenpunkte handelt (rund 0.727% des Datensatzes), lohnt sich dieser Aufwand praktisch nicht.

## Visualisierungen

### Numerische Features

![Pair Plot](figs/image-6.png)

Wir sehen, dass `bill_length_mm` allein schon eine ziemlich gute Trennung zwischen allen drei Arten ermöglicht. Bei den anderen Merkmalen, in denen keine `bill_length_mm` vorkommt, lässt sich zwar die Art `Gentoo` weiterhin gut unterscheiden, aber `Adelie` und `Chinstrap` überlappen stark.

Aus dieser Grafik können wir daher bereits vermuten, dass `bill_length_mm` ein besonders wichtiges Feature sein wird. Was wir an dieser Stelle jedoch noch nicht wissen, ist, mit welchen weiteren Merkmalen (`body_mass_g`, `flipper_length_mm`, `bill_depth_mm`) sich `bill_length_mm` am sinnvollsten kombinieren lässt, um eine möglichst robuste und gut generalisierbare Trennschärfe zu erreichen.

Auf der Hauptdiagonalen ist zu erkennen, dass die einzelnen Features innerhalb der jeweiligen Klassen näherungsweise normalverteilt sind.

Auch die kategorialen Variablen `sex` und `island` haben wir in dieser Darstellung noch nicht berücksichtigt.

### Kategorische Features

![Cross Tab](figs/image-7.png)

**Grafik links**: In der Stichprobe, wenn `island = torgersen`, wissen wir fast sicher, dass die Art `Adelie` ist. Außerdem sieht diese Variable ziemlich nützlich aus, aber wir wissen noch nicht, wie sie sich zusammen mit den anderen Variablen verhält.

**Grafik rechts**: Bei der Variable `sex` fällt auf, dass in dieser Stichprobe die Verteilungen für alle Arten ungefähr gleich sind. Jede Art hat ungefähr dieselbe Anzahl an männlichen und weiblichen Vögeln. Das bedeutet, dass `sex` keine Information zur Vorhersage von `species` enthält. Das werden wir uns später einmal in der Cramers V Matrix anschauen.

$$\Rightarrow P(\text{species} \mid \text{sex}) \approx P(\text{species})$$

**Achtung**: Nur weil diese Variable keinen direkten Zusammenhang mit der Zielvariable aufweist, bedeutet das nicht, dass sie in Kombination mit anderen Merkmalen nicht dennoch nützlich sein könnte. Wir haben ihren Einfluss auf andere Features noch nicht untersucht, daher bleibt sie vorerst im Datensatz.

### Numerisch x Kategorisch

![Multivariate](figs/image-9.png)

In dieser Visualisierung lassen sich insgesamt sechs Dimensionen gleichzeitig darstellen:

• `sex` in den Zeilen  
• `island` in den Spalten  
• `bill_length_mm` auf der x Achse  
• `bill_depth_mm` auf der y Achse  
• `body_mass_g` über die Punktgröße  
• `species` über die Farbe

Bereits auf den ersten Blick wird deutlich, dass `sex` praktisch keine zusätzliche Aussagekraft besitzt, selbst in Kombination mit mehreren anderen Variablen. Auch ein klarer Einfluss von `body_mass_g` lässt sich in diesen Plots kaum erkennen.

In diesen beiden Grafiken sind damit alle relevanten Features gemeinsam sichtbar. Es zeigt sich, dass in Anwesenheit von `bill_length_mm` und `bill_depth_mm` weder `sex` noch `body_mass_g` einen nennenswerten zusätzlichen Beitrag leisten. Wir lassen daher diese beiden Variablen weg und müssen uns nicht mehr um die fehlenden Werte in `sex` kümmern.

Für `island` ist die Situation weniger eindeutig. Zwar scheint ein Zusammenhang mit der Zielvariable zu bestehen, es bleibt jedoch offen, wie stark dieses Feature im Vergleich zu den morphologischen Merkmalen tatsächlich beiträgt.

*Dasselbe Muster zeigt sich auch bei `flipper_length_mm` (vgl. [notebook](notebook.ipynb)).*

Als potenzielle Features verbleiben daher `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` und `island`. Diese Merkmale erscheinen ausreichend, um die Klassifikation abzubilden. Welche davon letztlich verwendet werden und in welcher Kombination, wird im weiteren Verlauf entschieden.

![3D Plot](figs/image-12.png)  
*Interaktive Version: [Link](https://mib1213.github.io/MachineLearningLab/figs/scatter_3d_morphology.html)*

Zusätzlich haben wir versucht, die potenziellen Features in einem 3D Plot zu visualisieren. Dabei werden `bill_length_mm`, `bill_depth_mm` und `flipper_length_mm` gemeinsam betrachtet. In dieser Darstellung zeigt sich, dass `island` in Anwesenheit dieser drei morphologischen Merkmale kaum noch zusätzliche nützliche Informationen liefert.

Auf Basis dieser Beobachtung reduzieren sich die potenziellen Features weiter auf `bill_length_mm`, `bill_depth_mm` und `flipper_length_mm`.

## Korrelationsmatrix

![Pearson](figs/image-10.png)

Als Nächstes betrachten wir die Pearson Korrelationsmatrix der numerischen Features. Dabei wird schnell deutlich, warum `body_mass_g` in Anwesenheit anderer Merkmale kaum zusätzliche Vorhersagekraft liefert. Das Feature ist sehr stark mit `flipper_length_mm` korreliert und zeigt zudem eine deutliche Korrelation mit `bill_length_mm`. Unter diesen Umständen ist nicht zu erwarten, dass `body_mass_g` im Zusammenspiel mit diesen beiden Features noch wesentlich neue Informationen beisteuert.

Interessant ist außerdem die Korrelation zwischen `flipper_length_mm` und `bill_length_mm`, die moderat bis stark ausfällt. Das deutet darauf hin, dass beide Merkmale teilweise ähnliche Informationen tragen und gemeinsam nicht zwingend deutlich mehr beitragen als jeweils einzeln. Vor diesem Hintergrund erscheint die Kombination aus `bill_length_mm` und `bill_depth_mm` besonders sinnvoll, wenn eine Entscheidung auf nur zwei Features beschränkt werden muss.

## Feature Engineering

Ich habe außerdem zwei neue Features erzeugt: `bill_prop` (Verhältnis von Schnabellänge zu Schnabeltiefe) und `length_ratio` (Verhältnis von Schnabellänge zu Flipperlänge). Diese habe ich gemeinsam mit dem verbleibenden dritten Feature geplottet, um zu prüfen, ob sich die Klassen besser trennen lassen. Eine klare Verbesserung ist dabei jedoch nicht zu erkennen. Das deutet darauf hin, dass die neuen Features zwar inhaltlich sinnvoll erscheinen, aber keine zusätzlichen Informationen gegenüber den bestehenden Merkmalen liefern. *(vgl. [notebook](notebook.ipynb))*

## Outliers

Eine Analyse nach der 1,5 IQR Regel (Tukey) zeigt keine auffälligen Ausreißer in den betrachteten Features. *(vgl. [notebook](notebook.ipynb))*

# 3. Modellierung

## GNB

Der Gaussian Naive Bayes (GNB) Klassifikator kann in dieser Anwendung sinnvoll eingesetzt werden, aus mehreren Gründen:

1. Die verwendeten Features sind stetig
2. Die Feature Verteilungen gegeben der Klasse sind näherungsweise normalverteilt *(vgl. [notebook](notebook.ipynb), [3d Plot](https://mib1213.github.io/MachineLearningLab/figs/kde_3d_surface.html))*
3. Es handelt sich um ein White Box Modell, das transparent, leicht interpretierbar und in der Praxis einfach anzuwenden ist
4. Das Modell basiert auf soliden theoretischen Grundlagen aus der Wahrscheinlichkeitstheorie
5. Die Inferenz ist sehr schnell und effizient

Gleichzeitig besitzt Gaussian Naive Bayes auch einen zentralen Nachteil:

1. Es wird die naive Annahme getroffen, dass die Features gegeben der Klasse unabhängig voneinander sind

Zunächst verwenden wir die beiden Features `bill_length_mm` und `bill_depth_mm`, da sie sich in der explorativen Datenanalyse als besonders geeignet erwiesen haben.

Wir setzen ein a priori von 1/3 an. Das heißt, wir nehmen an, dass alle drei Klassen gleich wahrscheinlich sind, unabhängig von ihrer Häufigkeit im Datensatz. Die zugrunde liegende Annahme ist dabei, dass eine größere Anzahl an Messungen für eine bestimmte Art im Datensatz nicht zwangsläufig bedeutet, dass diese Art auch in der Realität häufiger vorkommt.

### LOOCV

Für die Evaluation verwende ich Leave One Out Kreuzvalidierung (LOOCV), da der Datensatz insgesamt relativ klein ist und LOOCV in diesem Fall einen sehr genauen Schätzer für die Generalisierungsleistung liefert.

**Ergebnis**:
* **train_accuracy**: 0.9450 ± 0.0007
* **validation_accuracy**: 0.9414 ± 0.2349

Die Accuracy auf den Trainingsdaten und in der LOOCV Validierung ist nahezu identisch und insgesamt relativ hoch. Gleichzeitig zeigt sich eine vergleichsweise hohe Standardabweichung der Validierungs Accuracy.

Daraus lassen sich folgende Schlüsse ziehen:

* Gaussian Naive Bayes passt gut zu den Daten
* Das Modell generalisiert stabil und zeigt kein starkes Overfitting
* Damit eignet sich GNB nicht nur als solides Basismodell, sondern hat grundsätzlich auch das Potenzial, in der Praxis eingesetzt zu werden

**Anmerkung zur Evaluation**: In LOOCV enthält jedes Test Fold genau einen Datenpunkt. Dadurch sind klassenweise Metriken wie Precision, Recall oder F1 pro Fold nur eingeschränkt interpretierbar. Aus diesem Grund fokussiere ich mich bei LOOCV auf die Accuracy und nutze für eine aussagekräftige klassenweise Bewertung zusätzlich Stratified K Fold mit Macro F1, Macro Precision, Macro Recall und Konfusionsmatrix.

### CV

![GNB Confusion Matrix](figs/image-14.png)

* **validation_accuracy**: 0.9454 ± 0.0371
* **validation_f1_score**: 0.9292 ± 0.0499
* **validation_precision_macro**: 0.9373 ± 0.0452
* **validation_recall_macro**: 0.9304 ± 0.0550

Da wir mit LOOCV kein detailliertes klassenweises Modellverhalten analysieren können, sondern im Wesentlichen nur die Accuracy betrachten, führen wir zusätzlich eine stratifizierte 10 Fold Kreuzvalidierung durch. Dadurch ist es möglich, Precision, Recall sowie die Konfusionsmatrix sinnvoll auszuwerten.

Precision und Recall liegen dabei sehr nahe beieinander. Das deutet darauf hin, dass das Modell keinen systematischen Bias in Richtung falsch positiver oder falsch negativer Vorhersagen aufweist. Die Konfusionsmatrix zeigt dass, die Klasse `Chinstrap` ist für das Modell am schwierigsten zu unterscheiden und wird häufiger mit den anderen Klassen verwechselt. Dagegen hat das Modell kaum Probleme, `Gentoo` und `Adelie` voneinander zu trennen.

### Entscheidungsgrenze

![GNB Decision Boundary](figs/image-13.png)

Hier ist die Entscheidungsgrenze des NB Modells inklusive Wahrscheinlichkeitskala dargestellt. Dabei sehen wir, dass Punkte, die weit von der Entscheidungsgrenze entfernt liegen, sehr eindeutig klassifiziert werden. Unsicherheiten treten fast ausschließlich in der Nähe der Entscheidungsgrenze auf.

Außerdem sehen wir, dass die Klasse `Chinstrap` teilweise mit den anderen Klassen überlappt. Besonders auffällig ist der Bereich etwa bei (`bill_length_mm` ≈ 55 bis 60, `bill_depth_mm` ≈ 17 bis 18). In diesem Bereich weist das Modell eine sehr hohe Vorhersagewahrscheinlichkeit für `Chinstrap` auf, teilweise nahe 1, obwohl dort vereinzelt sogar mehr Datenpunkte der Klasse `Gentoo` vorkommen.

Dieses Verhalten ist typisch für GNB. Da für jede Klasse eine Normalverteilung angenommen wird, erhalten Punkte, die weit vom jeweiligen Klassenmittelwert entfernt liegen, eine sehr geringe Wahrscheinlichkeit. Die in diesem Bereich liegenden `Gentoo` Punkte werden daher vom Modell praktisch ignoriert, da es unter der getroffenen Verteilungsannahme als sehr unwahrscheinlich gilt, dass ein `Gentoo` in diesem Bereich auftritt.

## kNN

Für ein Klassifikationsproblem mit wenigen numerischen Features ist es sinnvoll, kNN auszuprobieren. Obwohl das Modell sehr einfach ist, funktioniert es in vielen Fällen erstaunlich gut und kann mit deutlich komplexeren Modellen konkurrieren.

kNN macht dabei durchaus eine Annahme. Es geht davon aus, dass Datenpunkte mit dem gleichen Label im Feature Raum nah beieinander liegen. Anders gesagt: Ähnliche Features führen zu ähnlichen Labels. Damit diese Annahme Sinn ergibt, müssen Abstände zwischen Punkten aussagekräftig sein, weshalb eine sinnvolle Skalierung der Features wichtig ist.

Ein klarer Nachteil von kNN ist, dass es in höheren Dimensionen schlecht funktioniert (Curse of Dimensionality). In unserem Fall spielt das jedoch keine große Rolle, da wir nur mit wenigen Features arbeiten.

### Auswahl von k

Wir nehmen zunächst die drei potenziellen Features `bill_length_mm`, `bill_depth_mm` und `flipper_length_mm`. Diese skalieren wir mit `StandardScaler()` (Standardisierung auf 0 Mittelwert und 1 Standardabweichung). Anschließend suchen wir das beste *k* mithilfe von LOOCV im Bereich von 1 bis 50.

![k Neighbors](figs/kNeighbors.png)

Die Test Accuracy bleibt im Bereich zwischen k = 7 und k = 15 relativ stabil. Man würde hier eher k = 15 wählen, da ein etwas größeres k in der Regel besser generalisiert.

### LOOCV Feature Permutation <a id="loocv-feature-permutation-knn"></a>

* Features: `['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']`
  * **train_accuracy**: 0.9744 ± 0.0006
  * **validation_accuracy**: 0.9744 ± 0.1581
* Features: `['bill_length_mm', 'bill_depth_mm']`
  * **train_accuracy**: 0.9669 ± 0.0009
  * **validation_accuracy**: 0.9634 ± 0.1879
* Features: `['bill_length_mm', 'flipper_length_mm']`
  * **train_accuracy**: 0.9526 ± 0.0011
  * **validation_accuracy**: 0.9524 ± 0.2130

Jetzt schauen wir uns die LOOCV Performance von kNN für unterschiedliche Feature Mengen an, jeweils mit dem zuvor ausgewählten k. Dabei sieht man klar, dass die beste Performance mit allen drei Features erreicht wird und das Modell dabei auch gut generalisiert. Danach folgt die Kombination aus `bill_length_mm` und `bill_depth_mm`. Am schlechtesten schneidet die Kombination aus `bill_length_mm` und `flipper_length_mm` ab.

Die Entscheidung ist damit relativ eindeutig: Wenn die Priorität auf möglichst guter Vorhersage liegt, sollten alle drei Features verwendet werden. Wenn hingegen das Modellverhalten besser verstanden werden soll und dafür so wenige Features wie möglich genutzt werden sollen, ist die Teilmenge aus `bill_length_mm` und `bill_depth_mm` eine sinnvolle Wahl.

**Anmerkung:** Da ich k auf denselben Trainingsdaten auswähle, auf denen ich danach LOOCV berechne, können die Ergebnisse leicht optimistisch sein. Idealerweise würde man zuerst einen Train Validation Split machen, innerhalb der Trainingsdaten per stratifizierter Kreuzvalidierung k auswählen und erst danach mit LOOCV die Modellperformance zur Feature Auswahl anschauen.

### SHAP

Da kNN ein Black Box Modell ist, ist es sinnvoll, ein interpretierbares Verfahren zu verwenden, um das Verhalten der Features besser zu verstehen. In diesem Fall nutze ich SHAP, da das Verfahren modellagnostisch ist und sich daher gut mit kNN kombinieren lässt. Zudem erlaubt SHAP sowohl eine globale als auch eine lokale Analyse des Modellverhaltens.

**Adelie**  
![SHAP kNN Adelie](figs/shap_knn_adelie.png)

**Chinstrap**  
![SHAP kNN Adelie](figs/shap_knn_chinstrap.png)

**Gentoo**  
![SHAP kNN Adelie](figs/shap_knn_gentoo.png)

Hier sehen wir, dass die wichtigsten Features in allen drei Klassen entweder `bill_length_mm` oder `bill_depth_mm` sind, wenn wir das Modell mit allen drei Features verwenden. Das bestätigt unseren Eindruck aus der vorherigen Analyse, dass diese beiden Merkmale wichtiger sind als `flipper_length_mm`.

### CV

![kNN Confusion Matrix](figs/knn%20confusion%20matrix.png)

* **validation_accuracy**: 0.9636 ± 0.0360
* **validation_f1_macro**: 0.9506 ± 0.0478
* **validation_precision_macro**: 0.9689 ± 0.0369
* **validation_recall_macro**: 0.9417 ± 0.0536

Hier sieht man, dass das Modell insgesamt gut generalisiert. Die Validation Accuracy und die Precision sind sehr hoch, während der Recall etwas schlechter ausfällt. Das bedeutet, dass das Modell mehr False Negatives produziert.

Dieses Verhalten lässt sich sehr wahrscheinlich auf die Klasse `Chinstrap` zurückführen. Das Modell übersieht diese Klasse häufiger, als dass es sie fälschlicherweise erkennt. Konkret wurde `Chinstrap` 9 Mal übersehen und nur einmal falsch vorhergesagt. Aus der Analyse mit Naive Bayes wissen wir bereits, dass diese Klasse strukturell schwieriger zu unterscheiden ist, was unter anderem auch mit ihrem geringeren Vorkommen im Datensatz zusammenhängen kann (Klassenungleichgewicht).

Falls es besonders wichtig wäre, die Klasse `Chinstrap` nicht systematisch zu übersehen, ließe sich die Klassengewichtung in dem Modell entsprechend anpassen. Da kNN jedoch keine expliziten Klassengewichte unterstützt, müsste dieser Effekt über eine Gleichgewichtung der Klassen erzeugt werden, zum Beispiel durch Oversampling der `Chinstrap` Beobachtungen. Dadurch würden Precision und Recall tendenziell ausgeglichener, allerdings möglicherweise auf Kosten der Gesamtgenauigkeit und der Generalisierungsfähigkeit.

Da die aktuelle Priorität nicht auf der maximalen Erkennung von `Chinstrap` liegt, entscheide ich mich, mit dem bestehenden Modell unverändert weiterzuarbeiten.

### Entscheidungsgrenze

![kNN Decision Boundary](figs/knn_decision_boundary.png)

Hier sehen wir die Entscheidungsgrenze des Modells. Sie ist nicht so glatt wie bei GNB, schneidet insgesamt aber besser ab. Besonders deutlich wird das in den Bereichen etwa bei ([52 bis 60], [16 bis 18]) und ([45 bis 47], [20 bis 22]). Diese Punkte wurden von GNB praktisch komplett ignoriert, während kNN sie zumindest berücksichtigt, wenn auch mit geringerer Sicherheit.

Gleichzeitig sehen wir, dass das bereits diskutierte Problem mit False Negatives für die Klasse `Chinstrap` bei kNN etwas stärker ausgeprägt ist als bei GNB.

## Entscheidungsbaum

Es wäre eine Sünde, bei einem Klassifikationsproblem keine baumbasierten Modelle anzuschauen. Da unser Problem insgesamt relativ einfach ist, entscheide ich mich hier für das einfachste baumbasierte Modell, nämlich einen CART basierten Entscheidungsbaum.

Entscheidungsbäume haben einige klare Vorteile. Sie sind sehr gut interpretierbar, leicht nachvollziehbar und erlauben eine schnelle Inferenz. Gleichzeitig bringen sie aber auch typische Nachteile mit sich. Ohne Regularisierung neigen sie stark zum Overfitting (hohe Varianz), sie trennen immer nur entlang einer einzelnen Variable und erzeugen dadurch achsenparallele Entscheidungsgrenzen, und sie sind gierig. Das heißt, sie optimieren jeden Split lokal, was nicht zwangsläufig zu einer global optimalen Lösung führt.

```yaml
ccp_alpha: 0.0
class_weight: null
criterion: gini # ['gini', 'entropy'] 
max_depth: 4 # Baumtiefe
max_features: null # Anzahl zufälliger Features, die bei einem Split berücksichtigt werden
max_leaf_nodes: 10 # Anzahl Blätter
min_impurity_decrease: 0.0
min_samples_leaf: 1 # minimum ist 1
min_samples_split: 2 # minimum ist 2
min_weight_fraction_leaf: 0.0
monotonic_cst: null
random_state: 42
splitter: best # ['best' -> bestes Feature + bester Schwellen, 'random' -> zufälliges Feature + bester Schwellen]
```

Ein Entscheidungsbaum kann immer 100% Genauigkeit erreichen, gegeben genug Freiheitsgrade. Unser Ziel ist aber, die Lösung so einfach wie möglich zu behalten. Deswegen beschränken wir den Baum bis zur Tiefe 4 und auf maximal 10 Blätter. Diese Entscheidung ist quasi unsere obere Schranke für Komplexität. Wenn er mit diesem Komplexitätsniveau mit den anderen Modellen nicht konkurrieren kann, dann sind die anderen Alternativen bereits besser.

### LOOCV Feature Permutation <a id="loocv-feature-permutation-dtc"></a>

Wir verwenden alle 3 Featurekombinationen wie in kNN mit LOOCV:

* Features: `['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']`
  * **train_accuracy**: 0.9890 ± 0.0005
  * **validation_accuracy**: 0.9634 ± 0.1879

* Features: `['bill_length_mm', 'bill_depth_mm']`
  * **train_accuracy**: 0.9816 ± 0.0026
  * **validation_accuracy**: 0.9707 ± 0.1687

* Features: `['bill_length_mm', 'flipper_length_mm']`
  * **train_accuracy**: 0.9780 ± 0.0009
  * **validation_accuracy**: 0.9414 ± 0.2349

Die Validation Accuracy der Featurekombination `bill_length_mm` und `bill_depth_mm` ist am besten und sogar besser als die anderen Modelle, die wir bisher probiert haben, und das trotz der Einschränkung von Tiefe und Anzahl der Blätter.

### Visualisierung

![Tree](figs/tree_basic.svg)  
*Interaktive Version: [Link](https://mib1213.github.io/MachineLearningLab/figs/supertree.html)*

Wir sehen, dass der Baum ab der 4. Ebene im Wesentlichen versucht, nur noch einzelne Datenpunkte zu trennen.

An dieser Stelle sollten wir auf jeden Fall stoppen und anschließend überlegen, ob ein früheres Stoppen ebenfalls sinnvoll wäre und ob auf jeder Ebene tatsächlich alle Splits benötigt werden.

### Prunen

**Tiefe 2, Blatt 2**:  
Dieses Blatt muss nicht weiter gesplittet werden. Die Klasse ist hier eindeutig `Adelie`, und selbst bei weiteren Splits würden sich die Klassen nicht sinnvoll trennen. An dieser Stelle sollte man daher aufhören.

**Tiefe 2, Blatt 4**:  
Auch hier ist der Split nur bedingt sinnvoll, da die Klasse überwiegend `Chinstrap` ist und sich selbst mit weiteren Splits keine klare Trennung ergibt. Ob man hier stoppt oder noch weiter splittet, ist weniger eindeutig und hängt davon ab, ob die folgenden Splits tatsächlich neue Information liefern könnten.

**Tiefe 4, Blatt 7**:  
Im [Supertree](https://mib1213.github.io/MachineLearningLab/figs/supertree.html) sehen wir, dass es für den Baum sehr schwierig ist, diese Datenpunkte weiter zu trennen. Mit diesen beiden Features scheint es daher nicht sinnvoll zu sein, weiter zu versuchen. An dieser Stelle bietet es sich an zu prüfen, ob `flipper_length_mm` helfen kann, diese Punkte besser zu unterscheiden. Abgesehen von diesen etwa zehn Punkten lassen sich die Klassen sonst sehr gut trennen, sodass dieses zusätzliche Feature hier möglicherweise einen Mehrwert liefern könnte.

![Pairplot Chinstrap and Adelie](figs/pairplot%20chinstrap%20and%20adelie.png)

Hier sehen wir, dass sich diese etwa 10 Datenpunkte auch mithilfe von `flipper_length_mm` nicht sinnvoll trennen lassen. Wenn überhaupt, dann noch am ehesten über `bill_depth_mm`. Das heißt: Wenn `bill_length_mm` und `bill_depth_mm` hier bereits an ihre Grenzen stoßen, liefert `flipper_length_mm` ebenfalls keinen zusätzlichen Mehrwert.

Da der Split auf **Tiefe 2, Blatt 4** auch mit weiteren Splits (bis zu vier Ebenen tiefer) keine saubere Trennung erreicht, ist es sinnvoll, diesen Split komplett zu entfernen.

**Tiefe 1, Blatt 1**:  
Die letzte offene Entscheidung ist, ob dieser Split überhaupt benötigt wird. Er trennt lediglich drei `Gentoo` Punkte, was zunächst nach sehr wenig aussieht. Schaut man sich den Supertree jedoch genauer an, erkennt man, dass sich diese `Gentoo` Punkte klar und mit einer großen Lücke von den anderen Klassen unterscheiden. In diesem Fall ist der Split sinnvoll und führt nicht zu Overfitting, da hier eine systematische Trennung vorliegt.

### Manueller Baum

Wenn wir alle zusätzlichen Blätter entfernen, erhalten wir einen sehr einfachen und gut interpretierbaren Entscheidungsbaum:

* Wenn `bill_length_mm` ≤ 43.25 **und** `bill_depth_mm` > 14.8, dann `Adelie`
* Wenn `bill_length_mm` > 43.25 **und** `bill_depth_mm` > 16.45, dann `Chinstrap`
* Sonst `Gentoo`

#### Visualisierung

![Manual Tree](figs/manueller%20Baum.svg)

#### LOOCV

* **train_accuracy**: 0.9524 ± 0.0008
* **validation_accuracy**: 0.9524 ± 0.2130

Okay, wir verlieren etwa 2% Accuracy in der LOOCV im Vergleich zum nicht vereinfachten Baum mit vier Ebenen. Das ist durchaus ein Unterschied, aber diese Ergebnisse beziehen sich ausschließlich auf die Trainingsdaten. Unser eigentliches Ziel sind zukünftige, bisher ungesehene Daten.

Das vereinfachte Modell ist zwar etwas weniger genau, hat aber deutlich bessere Chancen, sauber zu generalisieren. Zusätzlich liegt der Fokus dieses Projekts klar auf Interpretierbarkeit. Der vereinfachte Baum ist wesentlich leichter und direkter zu verstehen als der vorherige, komplexere Baum.

Aus diesen Gründen entscheide ich mich bewusst dafür, mit dem vereinfachten Modell weiterzuarbeiten.

#### CV

![Confusion Matrix Manual](figs/confusion%20matrix%20man.png)

* **validation_accuracy**: 0.9526 ± 0.0327
* **validation_f1_macro**: 0.9404 ± 0.0429
* **validation_precision_macro**: 0.9427 ± 0.0388
* **validation_recall_macro**: 0.9459 ± 0.0464

Das Modell zeigt kein generelles Precision Recall Problem. Es ist lediglich, ähnlich wie bei GNB, insgesamt schwächer in der Vorhersage der Klasse `Chinstrap`.

#### Entscheidungsgrenze

![Decision Boundary Manual Tree](figs/decision%20boundary%20man.png)

In dieser Entscheidungsgrenze sehen wir im Grunde dasselbe Problem wie bei GNB, allerdings aus einem anderen Grund. Dort lag das Problem in der Annahme einer Normalverteilung. Hier liegt es an der Annahme achsenparalleler Entscheidungsgrenzen.

Wenn der Baum die Freiheit hätte, Entscheidungsgrenzen zu verwenden, die nicht achsenparallel sind, könnte er diese Klassen deutlich besser trennen. Da CART Entscheidungsbäume jedoch ausschließlich achsenparallele Splits erlauben, ist diese Einschränkung hier klar sichtbar.

# 4. Modellauswahl

## Entscheidungsgrenzen

![Decision Boundaries Comparison](figs/all%20decision%20boundaries.png)

## Konfusionsmatrizen

![Confusion Matrices](figs/confusionsmatrix.png)

kNN zeigt für die Klasse `Chinstrap` einen vergleichsweise schlechten Recall. In diesem Punkt schneidet sogar GNB besser ab, da GNB diese Klasse seltener übersieht.

## Benchmarking

![Benchmarking](figs/benchmarks.png)

kNN schneidet insgesamt am besten ab, gefolgt vom manuellen Entscheidungsbaum, während GNB die niedrigste Performance zeigt. GNB ist dabei relativ stark biased, was vor allem an der Annahme einer Normalverteilung liegt.

Der Entscheidungsbaum hat hauptsächlich einen anderen Nachteil. Innerhalb einer Region weist er allen Punkten die gleiche Wahrscheinlichkeit zu. Dadurch bekommen auch Punkte, die sehr weit von der Entscheidungsgrenze entfernt liegen und eigentlich eindeutig zu einer Klasse gehören, keine besonders hohe Wahrscheinlichkeit, wenn die gesamte Region als weniger zuverlässig eingestuft wird.

kNN hat diese beiden Nachteile nicht. Es findet eine sehr flexible Entscheidungsgrenze, ohne dabei stark zu overfitten, und ordnet jedem Punkt eine eigene Wahrscheinlichkeit zu, abhängig davon, wie nah er an der Entscheidungsgrenze liegt. Dadurch funktioniert kNN in diesem Fall sehr gut. Da wir außerdem nur mit zwei Features arbeiten, kann kNN hier sogar als eine Art Quasi White Box Modell betrachtet und relativ einfach interpretiert werden. Der eigentliche Nachteil von kNN ist die langsame Inferenz, da für jede Vorhersage alle Trainingspunkte betrachtet werden müssen. Das wird jedoch erst dann problematisch, wenn sehr viele Datenpunkte vorliegen. Unter diesen Bedingungen wäre kNN eine klare Wahl.

Man muss allerdings beachten, dass der Vergleich zwischen kNN und dem Entscheidungsbaum nicht ganz fair ist. kNN erhält hier maximale Freiheit, während der Entscheidungsbaum absichtlich manuell auf maximal drei Splits beschränkt wurde. Wir haben zuvor gesehen, dass ein Entscheidungsbaum mit vier Ebenen deutlich besser abschneidet als kNN. Wenn reine Performance das wichtigste Kriterium wäre, sollte daher eher ein tieferer Entscheidungsbaum verwendet werden.

Da ich mich in diesem Projekt jedoch bewusst für Einfachheit und Interpretierbarkeit entschieden habe, ist die Performance des manuellen Entscheidungsbaums dennoch sehr solide. Aus diesem Grund entscheide ich mich, weiterhin mit dem vereinfachten Entscheidungsbaum zu arbeiten.

## Bootstrapping

Zur zusätzlichen Einordnung der Modellperformance wurde einmalig das Bootstrap Verfahren mit 1000 Stichproben auf dem finalen Modell angewendet. Daraus ergeben sich folgende Schätzungen:

![Bootstrap Verfahren](figs/bootstrap.png)

Das Bootstrap Ergebnis dient ausschließlich zur Abschätzung der erwarteten Schwankungsbreite der Performance auf Basis der Trainingsdaten. Es wurde bewusst erst nach Abschluss aller Modellentscheidungen durchgeführt und nicht zur weiteren Modellwahl verwendet. In diesem Sinne stellt es keinen optimistischen Bias dar, sondern eine zusätzliche Plausibilitätsprüfung, bevor die Testdaten betrachtet werden.

# 5. Auswertung

## Evaluation

Jetzt ist der Moment der Wahrheit. Wir testen nun das final ausgewählte Modell auf den Testdaten. Als finales Modell habe ich den manuell konstruierten Entscheidungsbaum gewählt.

**Anmerkung:** Diese Testdaten wurden bisher nicht verwendet, weder für die explorative Datenanalyse noch für die Modellierung. Aus diesem Grund sollten sie einen guten Schätzer für die Performance auf zukünftigen, bisher ungesehenen Daten liefern.

| Klasse | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Adelie | 0.9667 | 0.9667 | 0.9667 | 30 |
| Chinstrap | 0.7500 | 0.8571 | 0.8000 | 14 |
| Gentoo | 0.9565 | 0.8800 | 0.9167 | 25 |
| **Accuracy** |  |  | **0.9130** | **69** |
| Macro Avg | 0.8911 | 0.9013 | 0.8944 | 69 |
| Weighted Avg | 0.9190 | 0.9130 | 0.9147 | 69 |

![Confusion Matrix Test](figs/confusion%20matrix%20test.png)

Dieses Ergebnis habe ich in dieser Form bereits erwartet. Mit einem voll ausgebauten Entscheidungsbaum oder mit kNN hätten sich vermutlich bessere Ergebnisse erzielen lassen. Interessant ist zudem, dass die Test Accuracy leicht unterhalb des Bootstrap Intervalls liegt, was zunächst unerwartet war. Bei der geringen Größe des Testsets ist dieses Ergebnis jedoch gut durch zufällige Schwankungen erklärbar. Dennoch bin ich mit dem gewählten Modell zufrieden, da es eine sehr schnelle und klare Interpretierbarkeit ermöglicht.


Das folgt derselben Idee wie bei klinischen Score Modellen. Die Entscheidungslogik ist vollständig transparent, lokal interpretierbar und erlaubt es, Modellvorhersagen kritisch nachzuvollziehen, anstatt sie ungeprüft zu akzeptieren. Genau aus diesem Grund sind solche Modelle in der klinischen Praxis weit verbreitet und etabliert.

![Final Decision Boundary](figs/final%20decision%20boundary.png)

Die Testdaten zeigen genau an den gleichen Stellen Probleme, an denen wir auch bereits bei den Trainingsdaten Schwierigkeiten gesehen haben. Das ist ein klares Zeichen für Underfitting. Das Modell könnte diese Muster grundsätzlich lernen, wenn wir es weniger stark eingeschränkt hätten.

Wir haben uns jedoch bewusst für diese Einschränkung entschieden. Der Fokus liegt nicht auf maximaler Performance, sondern auf Einfachheit, Stabilität und guter Interpretierbarkeit.

## Interpretation

Wir schauen uns jetzt genauer an, warum das Modell bestimmte Fehler macht. Zum Beispiel: Warum sagt es 5 Mal `Chinstrap`, obwohl die tatsächliche Klasse `Adelie` ist?

![SHAP False Prediction](figs/shap%20false%20prediction.png)  
![SHAP Right Prediction](figs/shap%20right%20prediction.png)

Wir sehen hier, dass `bill_length_mm` eindeutig für diese falsche Vorhersage verantwortlich ist. Konkret reduziert der Wert dieses Features im Modell die Wahrscheinlichkeit für die korrekte Klasse `Adelie` von etwa 48% auf nahezu 0%.

![Prediction Box Plots](figs/prediction%20boxplots.png)

Hier sehen wir, dass der Wert von `bill_length_mm` deutlich näher am `Chinstrap` Cluster liegt als am `Adelie` Cluster. In diesem Fall hat das Modell also tatsächlich sinnvoll entschieden, da wir intuitiv vermutlich zur gleichen Einschätzung gekommen wären.

Man könnte überlegen, ob ein zusätzliches Feature wie `flipper_length_mm` dieses Problem lösen könnte. Im rechten Plot sehen wir jedoch, dass auch dieses Feature näher am `Chinstrap` Cluster liegt als am `Adelie` Cluster. Das könnte sogar ein Hinweis darauf sein, dass das Label dieses Datenpunkts möglicherweise falsch ist.

# 6. Fazit

Ziel dieser Arbeit war es, ein Klassifikationsmodell für den Palmer Penguins Datensatz zu entwickeln, das nicht nur gute Vorhersagen liefert, sondern vor allem gut interpretierbar ist. Der Fokus lag dabei absichtlich nicht auf maximaler Accuracy, sondern auf einem klar nachvollziehbaren und verständlichen Modellverhalten.

Die wenigen fehlenden Werte in relevanten Features wurden gezielt entfernt, da es sich insgesamt nur um sehr wenige Datenpunkte handelte. Für diese Entscheidung habe ich mich bewusst entschieden, da mir eine saubere Modellierung wichtiger war als für jeden einzelnen Datenpunkt eine Vorhersage zu erzwingen. Auch im Testset hätte ich fehlende Werte auf die gleiche Weise behandelt. Ausgehend von einer ausführlichen explorativen Datenanalyse habe ich anschließend verschiedene Feature Kombinationen untersucht und drei unterschiedliche Modellfamilien miteinander verglichen, nämlich Gaussian Naive Bayes, kNN und Entscheidungsbäume.

Dabei zeigte sich, dass die Merkmale `bill_length_mm` und `bill_depth_mm` den größten Beitrag zur Trennung der Klassen leisten. Weitere Features wie `flipper_length_mm`, `body_mass_g`, `sex` oder `island` lieferten entweder keine zusätzliche Information oder führten zu einer schlechteren Generalisierung.

Im Modellvergleich erzielte kNN, abgesehen von einem Entscheidungsbaum mit vier Ebenen, die beste Performance. Gleichzeitig zeigte kNN jedoch Schwächen beim Recall der Klasse `Chinstrap` und ist in der Inferenz in der Regel vergleichsweise langsam. Gaussian Naive Bayes war stark durch seine Verteilungsannahmen eingeschränkt und schnitt insbesondere bei überlappenden Klassen schlechter ab. Ein Entscheidungsbaum mit vier Ebenen funktionierte zwar am besten, wurde jedoch bewusst auf maximal drei Splits beschränkt. Dadurch entstand ein manuell vereinfachter Entscheidungsbaum, der deutlich transparenter, stabiler und einfacher zu interpretieren ist.

Das finale Modell besteht aus wenigen, klaren Entscheidungsregeln und ist damit leicht verständlich und direkt erklärbar. Die Evaluation auf dem zuvor vollständig ungenutzten Testset bestätigte die erwartete Performance und zeigte, dass die Fehlerstruktur konsistent mit den Beobachtungen aus den Trainingsdaten ist. Insgesamt erfüllt das gewählte Modell das ursprüngliche Ziel dieser Arbeit sehr gut.

# 7. Reflexion

Rückblickend gibt es einige Strategien, die ich hätte anders oder konsequenter umsetzen können und die theoretisch zu besseren Ergebnissen geführt hätten.

* Eine naheliegende Möglichkeit wäre gewesen, die Klassen durch ein Oversampling Verfahren auszugleichen. Dadurch hätte man die Probleme mit der seltenen Klasse `Chinstrap` vermutlich deutlich reduzieren können, insbesondere im Hinblick auf den Recall
* Auch der Umgang mit fehlenden Werten hätte robuster gestaltet werden können, etwa durch eine explizite Fallback Logik oder eine getrennte Inferenz Pipeline für unvollständige Beobachtungen
* Ein weiterer Punkt betrifft den Umgang mit dem Feature `island`. Statt viel Zeit in die detaillierte Analyse dieses Merkmals zu investieren, hätte ich auch von Anfang an stärker von einer möglichen kontextuellen Verzerrung der Datenerhebung ausgehen und das Feature frühzeitig aus dem Modell entfernen können. Ich habe mich bewusst dagegen entschieden und `island` genauer untersucht, um die zugrunde liegenden Annahmen transparent zu machen. Am Ende zeigte sich jedoch, dass das Feature keinen ausreichenden zusätzlichen Nutzen bringt und daher ohnehin verworfen wurde
* Auch beim manuellen Entscheidungsbaum wäre eine andere Strategie möglich gewesen. An einigen Stellen wurde sichtbar, dass bestimmte Splits lokal nicht optimal waren und dass zusätzliche Splits die Trennung einzelner Datenpunkte verbessert hätten, ohne zwangsläufig die Generalisierung stark zu verschlechtern. Mit einem weniger stark eingeschränkten Baum wären auf dem Testset vermutlich bessere Ergebnisse erzielbar gewesen

Diese Einschränkungen habe ich jedoch bereits während der Modellierung explizit in Kauf genommen und an den jeweiligen Stellen begründet. Ziel dieser Arbeit war es nicht, das leistungsstärkste Modell zu finden, sondern ein Modell zu entwickeln, dessen Entscheidungslogik klar, stabil und nachvollziehbar ist. Die getroffenen Entscheidungen spiegeln diesen Fokus wider, auch wenn dadurch gezielt auf einen Teil der maximal möglichen Performance verzichtet wurde.
