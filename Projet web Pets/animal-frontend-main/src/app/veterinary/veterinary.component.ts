import { Component } from '@angular/core';
@Component({
  selector: 'app-veterinary',
  templateUrl: './veterinary.component.html',
  styleUrls: ['./veterinary.component.scss']
})
export class VeterinaryComponent {
  veterinaries=[
    {name:"Hana najeh",country:"Tunisia",address:"Ariana",phone:"56512342",img:"../../assets/vetr1.jpg"},
    {name:"Sarra saroour",country:"Tunisia",address:"Ariana",phone:"29895478",img:"../../assets/vetr2.jpg"},
    {name:"Si Kammoun",country:"Tunisia",address:"Ariana",phone:"97892145",img:"../../assets/vetr3.jpg"},
    {name:"Yass",country:"Tunisia",address:"Ariana",phone:"31568988",img:"../../assets/vetr5.png"},
    {name:"Wee wee",country:"Tunisia",address:"Ariana",phone:"31568988",img:"../../assets/vetr2.jpg"},
    {name:"Barhoum",country:"Tunisia",address:"Ariana",phone:"31568988",img:"../../assets/vetr1.jpg"},

  ]
}
