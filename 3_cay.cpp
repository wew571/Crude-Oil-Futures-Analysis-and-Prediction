#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <thread>  
#include <chrono>
#include <random>
#include <ctime>
#include <limits>
using namespace std;

class Card {
private:
    string suit, rank;
public:
    Card(string suit, string rank) : suit(suit), rank(rank) {}

    string getRank() const {
         return rank; 
    }

    string getSuit() const { 
        return suit; 
    }

    int getPoint() const {
        if (rank == "A") {
            return 1;
        }
        if (rank == "J" || rank == "Q" || rank == "K" || rank == "10") {
            return 10;
        }    
        return stoi(rank);
    }

    int getSuitPriority() const { 
        if (suit == "Bich"){
            return 4;
        }    
        if (suit == "Co"){
            return 3;
        }    
        if (suit == "Ro"){
            return 2; 
        }    
        return 1; 
    }

    void Display() const { 
        cout << rank << " " << suit;
    }
};

class Deck {
    private:
        vector<Card> cards;
    public:
        Deck(){
            createDeck();
        }

        void createDeck(){
            string const suits[] = {"Co", "Ro", "Tep", "Bich"};
            string const ranks[] = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};
            for (const string &suit : suits) {
                for (const string &rank : ranks) {
                    cards.emplace_back(suit, rank);
                }
            }
            Shuffle();
        }

        void Shuffle(){
            random_device rd;
            mt19937 g(rd());
            shuffle(cards.begin(), cards.end(), g);
        }

        
        Card draw(){
            if (cards.empty()) {
                createDeck();
            }
            Card topCard = cards.back();
            cards.pop_back();
            return topCard;
        }
};

class Player {
    protected:
        string name;
        vector<Card> hand;
        int money, bet;
        bool isBot;
    public:
        Player(string name, int money = 1000, bool isBot = false) : name(name), money(money), bet(0), isBot(isBot) {}
    
        void resetHand(){
            hand.clear();  
        }
    
        void placeBet(){
            if (isBot) {
                bet = rand() % (money / 2) + 10;  
                cout << name << " Bets : " << bet << "$\n";
            } else {
                cout << name << " Is Having " << money << "$. Enter Your Bet : ";

                while (true) {
                    string input;
                    cin >> input;
        
                    
                    bool isValidNumber = true;
                    for (char c : input) {
                        if (!isdigit(c)) {
                            isValidNumber = false;
                            break;
                        }
                    }
        
                    if (!isValidNumber) {
                        cout << "Invalid Bet! Please Enter A Number: ";
                        cin.clear();
                        cin.ignore(numeric_limits<streamsize>::max(), '\n');
                        continue;
                    }
        
                    bet = stoi(input);  
                    if (bet > money || bet <= 0) {
                        cout << "Invalid Bet! Please Re-Enter: ";
                        continue;
                    }
                    break;
                }
            }
            money -= bet;
        }
    
        void draw(Deck &deck) {
            hand.push_back(deck.draw());
        }
    
        void showHand() const {
            cout << name << " Flip : ";
            for (const Card &card : hand) {
                card.Display();
                cout << " | ";
            }
            cout << endl;
        }
    
        int getTotalPoints() const {
            int total = 0;
            for (const Card &card : hand) {
                total += card.getPoint();
            }
            if (total == 10 || total == 20 || total == 30){
                return 10;
            } 
            return total % 10;
        }
    
        Card getHighestCard() const {
            return *std::max_element(hand.begin(), hand.end(),
                [](const Card &a, const Card &b) {
                    if (a.getSuitPriority() == b.getSuitPriority())
                        return a.getRank() < b.getRank(); // nếu cùng chất thì so sánh giá trị
                    return a.getSuitPriority() < b.getSuitPriority();
                });
        }

    
        string getName() const {
            return name; 
        }

        int getBet() const {
            return bet; 
        }

        void addMoney(int amount) {
            money += amount; 
        }
        int getMoney() const {
            return money; 
        }

        bool isBankrupt() const {
            return money <= 0;
        }
        
    };    

class Game {
private:
    Deck deck;
    vector<Player> players;
public:
    Game( int numberPlayers) {
        string playerName;
        cout << "Enter Your Name : ";
        cin >> playerName;
        players.emplace_back(playerName, 1000, false);  

        for (int i = 1; i <= numberPlayers; i++) {
            players.emplace_back("Bot " + to_string(i), 1000, true);
        }
    }

    void startGame() {
        deck.Shuffle();
        
        for (Player &player : players) {
            player.resetHand();  
            player.placeBet();  
        }
    
        for (int i = 0; i < 3; i++) {
            for (Player &player : players) {
                player.draw(deck);
            }
        }
    }

    void revealHands() {
        for (Player &player : players) {
            cout << player.getName() << " Waiting...\n";
            this_thread::sleep_for(chrono::seconds(player.getName() == players[0].getName() ? 0 : 2));  
            player.showHand();
            cout << "Points: " << player.getTotalPoints() << endl;
            cout << "----------------------\n";
        }
    }

    void findWinner() {
        Player *winner = &players[0];

        for (size_t i = 1; i < players.size(); i++) {
            int p1 = winner->getTotalPoints();
            int p2 = players[i].getTotalPoints();
            
            if (p2 > p1 || (p2 == p1 && players[i].getHighestCard().getSuitPriority() > winner->getHighestCard().getSuitPriority())) {
                winner = &players[i];
            }
        }

        int totalBet = 0;
        for (const Player &player : players) {
            totalBet += player.getBet();
        }

        winner->addMoney(totalBet);
        cout << "Player " << winner->getName() << " Win With " << winner->getTotalPoints() << " Points!\n";
        cout << winner->getName() << " Obtained " << totalBet << "$,        Current Player Win Have Money : " << winner->getMoney() << "$\n";
    }

    void checkPlayers() {
        if (players[0].isBankrupt()) {
            cout << " You Run Out Of Money! Game Over.\n";
            exit(0);
        }
    
    players.erase(
        std::remove_if(players.begin(), players.end(), [](const Player &p) {
            return p.isBankrupt();
        }),
        players.end()
);

    
        if (players.size() == 1) {
            cout << " " << players[0].getName() << " The Last One With Money , Game Over !.\n";
            exit(0);
        }
    }
    
};

int main() {
    srand(time(0));  
    cout << "How Many Players Do You Want ?" << endl;
    int numberPlayer ;
    while(true){
        string input;
        cin >> input;
        bool isValidNumberPlayer = true;

        for (char c : input){
            if ( !isdigit(c)){
                isValidNumberPlayer  = false;
            }
        }
        
        if (!isValidNumberPlayer){
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max() , '\n');
            cout << "Number Player Is Invaid ! Please Re-Enter Number Player" << endl;
            continue;
        }
        
        numberPlayer = stoi(input);
        if ( numberPlayer < 0 || numberPlayer > 13){
            cout << "Number Player Is Invaid ! Please Re-Enter Number Player" << endl;
            continue;
        }
        break;
    }
    Game game(numberPlayer);

    while (true) {
        game.startGame();
        game.revealHands();
        game.findWinner();
        game.checkPlayers();
        int cont;

        while( true){
        cout << "Do You Want Countinue ? " << endl  << "1 Countinue Game" << endl << "2 Quit Game" << endl;
        cout << "Enter Your Select ";
        cin >> cont;
        if (cin.fail()){
            cin.clear();
            cin.ignore(numeric_limits<streamsize>:: max() , '\n'); 
            cout << "Enter Ur Select Again!\n";
            continue;
        }
        if (cont == 2 || cont == 1){
            break;
            }
        }
        if (cont == 2){
            break;
        }
    }
    return 0;
}
//g++ -std=c++17 -o main 3_cay.cpp
//./main
