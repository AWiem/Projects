import { Component } from '@angular/core';

@Component({
  selector: 'app-find-a-keeper',
  templateUrl: './find-a-keeper.component.html',
  styleUrls: ['./find-a-keeper.component.scss']
})
export class FindAKeeperComponent {
  keepers=[
    {name:"Ahmed Mastouri",country:"Tunisia",address:"Ariana",phone:"56512342",img:"../../assets/avatar1.jpg"},
    {name:"Salma Maalej",country:"Tunisia",address:"Bardo",phone:"29895478",img:"../../assets/avatar2.jpg"},
    {name:"Karam Banen",country:"Tunisia",address:"Bizerta",phone:"97892145",img:"../../assets/avatar3.jpg"},
    {name:"Yasmine Missoui",country:"Tunisia",address:"Hammamet",phone:"31568988",img:"../../assets/avatar5.jpg"},
    {name:"Wided Boukthir",country:"Tunisia",address:"Sfax",phone:"31568988",img:"../../assets/avatar4.jpg"},

  ]
}

