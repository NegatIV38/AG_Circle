#define OMP //active le multi-threading
#include <SFML/Graphics.hpp>
#include <SFML/System/Time.hpp>
#include <iostream>
#include <cmath>
#include <Windows.h>
#include <ctime>
#include <algorithm>
#include <memory>
#include <omp.h>

#define ARG

#ifndef ARG
#define NB_POINTS 5
#define NB_POP 35
#define PRECISION 20
#endif
#ifdef ARG
#define NB_POINTS atoi(argv[1])
#define NB_POP atoi(argv[2])
#define PRECISION atoi(argv[3])
#endif

#define PI 3.14159265359

#define SOMME

//#define OPT //active la multiplication des threads

using namespace sf;
using namespace std;

typedef struct{
	vector<float> fitnessParPoints;
	float fitnessMoyen;
	vector<shared_ptr<CircleShape>> points;
	vector<shared_ptr<Vector2f>> positions;
} individu;

void evalFitnessPoints(individu* indiv, vector<CircleShape*>* pts){
	for (int i = 0; i < indiv->positions.size(); i++){
		int a = pow(pts->at(i)->getPosition().x - indiv->positions.at(i)->x, 2) + pow(pts->at(i)->getPosition().y - indiv->positions.at(i)->y, 2);
		indiv->fitnessParPoints.at(i) = sqrt(a);
	}
}

void evalFitnessMoyen(individu* indiv){
	int somme = 0;
	for (int i = 0; i < indiv->fitnessParPoints.size(); i++){
		somme += indiv->fitnessParPoints.at(i);
	}
#ifdef MOYENNE
	indiv->fitnessMoyen = somme / indiv->fitnessParPoints.size();
#endif
#ifdef SOMME
	indiv->fitnessMoyen = somme;
#endif
}

void evalFitnessPop(vector<individu>* pop, vector<CircleShape*>* pts){
#ifdef OMP
#pragma omp parallel for
#endif
	for (int i = 0; i < pop->size(); i++){
		evalFitnessPoints(&pop->at(i), pts);
		evalFitnessMoyen(&pop->at(i));
	}
}

void affIndiv(individu* indiv, RenderWindow* win){
	Time t = seconds(0.1f);
#ifdef OMP
#pragma omp parallel shared(win,indiv)	
	win->setActive(false);
#pragma omp master 
	win->setActive(true);
#pragma omp for
#endif
	for (int i = 0; i < indiv->points.size(); i++){

		//win->display();
		//sf::sleep(t);
		win->draw(*indiv->points.at(i));
	}
}

void affPop(vector<individu>* pop, RenderWindow* win){
	for (int i = 0; i < pop->size(); i++){
		affIndiv(&pop->at(i), win);
	}
}

void initPos(individu* indiv, RenderWindow* win){
	for (int i = 0; i < indiv->positions.size(); i++){
		indiv->positions.at(i).reset(new Vector2f(rand() % win->getSize().x, rand() % win->getSize().y));
	}
}

void initCir(individu* indiv){
	Color c((rand() % 200) + 25, (rand() % 200) + 25, (rand() % 200) + 25);
	for (int i = 0; i < indiv->points.size(); i++){
		indiv->points.at(i).reset(new CircleShape(5));
		indiv->points.at(i)->setOrigin(indiv->points.at(i)->getRadius(), indiv->points.at(i)->getRadius());
		indiv->points.at(i)->setFillColor(c);
		indiv->points.at(i)->setPosition(*indiv->positions.at(i));
	}
}

void initPop(vector<individu>* pop, RenderWindow* win,char* argv[]){
	for (int i = 0; i < pop->size(); i++){
		pop->at(i).fitnessParPoints.resize(NB_POINTS);
		pop->at(i).points.resize(NB_POINTS);
		pop->at(i).positions.resize(NB_POINTS);
		initPos(&pop->at(i), win);
		initCir(&pop->at(i));
	}
}

void initPts(std::vector<CircleShape*>* pts, CircleShape& cir, char* argv[]){
	for (int i = 0; i < NB_POINTS; i++){
		pts->at(i) = new CircleShape(5);
		pts->at(i)->setOrigin(pts->at(i)->getRadius(), pts->at(i)->getRadius());
		pts->at(i)->setFillColor(Color::Green);
		pts->at(i)->setPosition(std::cos(i*(2 * PI / NB_POINTS)) * cir.getRadius() + cir.getPosition().x, std::sin(i*(2 * PI / NB_POINTS)) * cir.getRadius() + cir.getPosition().y);
	}
}

void afficherPtsControle(RenderWindow* win, vector<CircleShape*>* pts){
	for (int i = 0; i < pts->size(); i++){
		win->draw(*pts->at(i));
	}

}

void setCircle(CircleShape* cir, RenderWindow* win){
	int radius = cir->getRadius();
	cir->setOrigin(radius, radius);
	cir->setOutlineColor(Color::Green);
	cir->setOutlineThickness(2.0f);
	cir->setFillColor(Color::Black);
	cir->setPosition(win->getSize().x / 2, win->getSize().y / 2);
}

bool trier_fitness(individu x, individu y){
	return (x.fitnessMoyen < y.fitnessMoyen);
}

void trier_pop(vector<individu>* pop){
	sort(pop->begin(), pop->end(), trier_fitness);
}

void calcProba(vector<float>* proba, vector<individu>* pop){
	proba->at(0) = pop->at(0).fitnessMoyen;
	for (int i = 1; i < pop->size(); i++){
		proba->at(i) = pop->at(i).fitnessMoyen + proba->at(i - 1);
	}
	for (int i = 0; i < proba->size(); i++){
		proba->at(i) = proba->at(proba->size() - 1) - proba->at(i);
	}
}

void selectParents(vector<individu>* pop, pair<int, int>* p, vector<float>* proba, char* argv[]){
	int r;
	r = rand() % int(ceil(proba->at(0)));
	for (int i = 0; i < NB_POP; i++){
		if (r < proba->at(i)){
			p->first = i;
			i = NB_POP;
		}
	}
	int j = 0;
	j = -1;
	r = rand() % int(ceil(proba->at(0)));
	for (int i = 0; i < NB_POP; i++){
		if (r < proba->at(i)){
			p->second = i;
			i = NB_POP;
		}
		j++;
	}
	//cout << "p:" << p->first << " m:" << p->second << endl;
}

void mutation(individu* indiv, RenderWindow* win, individu * best, char* argv[]){
	//indiv->positions.at(rand() % (indiv->positions.size()))->x = rand() % win->getSize().x;
	//indiv->positions.at(rand() % (indiv->positions.size()))->y = rand() % win->getSize().y;
	int r = rand() % (indiv->positions.size());
	indiv->positions.at(r).reset(new Vector2f(best->positions.at(r)->x + ((rand() % PRECISION) - PRECISION / 2),
		best->positions.at(r)->y + ((rand() % PRECISION) - PRECISION / 2)));

}

individu child(vector<individu>* pop, vector<CircleShape*>* pts, RenderWindow* win, vector<float>* proba, individu* best, char* argv[]){
	pair<int, int> p;
	selectParents(pop, &p, proba,argv);
	individu c;
	c.fitnessParPoints.resize(NB_POINTS);
	c.points.resize(NB_POINTS);
	c.positions.resize(NB_POINTS);
	int r = (rand() % (pop->at(p.first).positions.size()));
	for (int i = 0; i < r; i++){
		c.positions.at(i) = pop->at(p.first).positions.at(i);
	}
	for (int i = r; i < pop->at(p.second).positions.size(); i++){
		c.positions.at(i) = pop->at(p.second).positions.at(i);
	}
	mutation(&c, win, best,argv);
	initCir(&c);
	evalFitnessPoints(&c, pts);
	evalFitnessMoyen(&c);
	return c;
}

void new_pop(vector<individu>* pop, vector<CircleShape*>* pts, RenderWindow* win, vector<float>* proba, individu* best, char* argv[]){
	vector<individu> buffer(NB_POP);
	double t = time(NULL);
#ifdef OMP
#pragma omp parallel for
#endif
	for (int i = 0; i < NB_POP; i++){
		buffer.at(i) = (child(pop, pts, win, proba, best,argv));
	}
	//deletePop(pop,best);
	*pop = buffer;
	//cout << "time Newpop : " << time(NULL) - t << endl; //between 2-4s for 200 people
}

void reinjection(vector<individu>* pop, individu* best){
	int imin = 0;
	for (int i = 1; i < pop->size(); i++){
		if (pop->at(i).fitnessMoyen > pop->at(imin).fitnessMoyen){
			imin = i;
		}
	}
	pop->at(imin) = *best;
}

void setColorIndiv(individu* indiv){
	for (int i = 0; i < indiv->points.size(); i++){
		indiv->points.at(i)->setFillColor(Color::Red);
	}
}

void affLiens(RenderWindow* win, individu* indiv, char* argv[]){
	for (int i = 0; i < NB_POINTS - 1; i++){
		Vertex line[] = { Vertex(Vector2f(indiv->points.at(i)->getPosition().x, indiv->points.at(i)->getPosition().y)), Vertex(Vector2f(indiv->points.at(i + 1)->getPosition().x, indiv->points.at(i + 1)->getPosition().y)) };
		win->draw(line, 2, Lines);
	}
	Vertex line[] = { Vertex(Vector2f(indiv->points.at(NB_POINTS - 1)->getPosition().x, indiv->points.at(NB_POINTS - 1)->getPosition().y)), Vertex(Vector2f(indiv->points.at(0)->getPosition().x, indiv->points.at(0)->getPosition().y)) };
	win->draw(line, 2, Lines);
}

int main(int argc , char* argv[])
{
	cout << "argc : " << argc << endl;
	srand(time(NULL));
	int t0 = time(NULL);
	Font font;
	font.loadFromFile("arial.ttf");
	sf::Text text;
	text.setFont(font);
	text.setString("0");
	text.setCharacterSize(24);
	text.setColor(sf::Color::White);
	text.setStyle(sf::Text::Bold | sf::Text::Underlined);
	RenderWindow window(VideoMode(600, 600), "SFML works!");
	float radius = 128.f;
	CircleShape shape(radius);
	setCircle(&shape, &window);
	vector<CircleShape*> ptsControle(NB_POINTS);
	initPts(&ptsControle, shape,argv);
	vector<individu> population(NB_POP);
	initPop(&population, &window,argv);
	vector<float> proba(NB_POP);
	float meilleur = NB_POINTS * 10000;
	individu best;
	int generation = 0;
	best.fitnessParPoints.resize(NB_POINTS);
	best.points.resize(NB_POINTS);
	best.positions.resize(NB_POINTS);
	initPos(&best, &window);
	initCir(&best);
	evalFitnessPoints(&best, &ptsControle);
	evalFitnessMoyen(&best);
#ifdef OMP
	int nthreads = 0;
	int threadid = 0;
#ifdef OPT
	omp_set_num_threads(NB_POP / omp_get_num_procs());
#endif

#pragma omp parallel private(threadid)             
	{
#pragma omp master                              
		{
			nthreads = omp_get_num_threads();
			cout << endl << nthreads << " thread(s) available for computation" << endl;
		}

#pragma omp barrier                             

		threadid = omp_get_thread_num();
#pragma omp critical                            
		{
			cout << "Thread " << threadid << " is ready for computation" << endl;
		}
	}
#endif
#ifndef OMP
	omp_set_num_threads(1);
#endif
	while (window.isOpen() && meilleur > 0.1f)
	{
		Event event;
		while (window.pollEvent(event))
		{
			if (event.type == Event::Closed)
				window.close();
		}

		window.clear();
		window.draw(text);
		window.draw(shape);
		afficherPtsControle(&window, &ptsControle);
		//affPop(&population, &window);
		setColorIndiv(&best);
		affIndiv(&best, &window);
		affLiens(&window, &best,argv);
		evalFitnessPop(&population, &ptsControle);
		reinjection(&population, &best);
		trier_pop(&population);
		if (population.at(0).fitnessMoyen < meilleur){
			//deleteBest(&best);
			best = population.at(0);
			meilleur = population.at(0).fitnessMoyen;
			cout << "[" << meilleur << "]" << endl;
		}
		calcProba(&proba, &population);
		new_pop(&population, &ptsControle, &window, &proba, &best,argv);
		generation++;
		if (generation % 50 == 0){
#pragma omp parallel 
			{
#pragma omp master 
				{
					cout << "time : " << time(NULL) - t0 << " Threads actifs : " << omp_get_num_threads() << endl;
				}
			}
		}
		text.setString(to_string(generation));
		window.display();
	}
	setColorIndiv(&best);
	affIndiv(&best, &window);
	affLiens(&window, &best,argv);
	window.display();
	system("PAUSE");
	return 0;
}