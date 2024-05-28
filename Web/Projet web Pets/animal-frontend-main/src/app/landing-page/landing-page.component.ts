import { Component } from '@angular/core';

@Component({
  selector: 'app-landing-page',
  templateUrl: './landing-page.component.html',
  styleUrls: ['./landing-page.component.scss']
})
export class LandingPageComponent {
  cards1=[
    {id:1,title:"Community",content:"Join like minded people in your local area who are part of the largest dog-loving community"},
    {id:2,title:"Safety",content:"All Premium members take our safety checks, and Premium membership includes insurance and access to a Vet Line"},
    {id:3,title:"Feel good",content:"Our members sign up to make another person happy by sharing the love of dogs"}
  ]
  cards2=[1,2,3]

  cardshowitworks=[
    {id:1,title:"Create your free profile",content:"Create a profile for you and your pets for the best experience"},
    {id:2,title:"Find a veterinary",content:"Find the best veterinaries in your surrounding areas"},
    {id:3,title:"Meet a house sitter",content:"Start searching for friendly and trustworthy local dogs sitters."},
    {id:4,title:"Adopt a pet",content:"it's the right place to find your future companion"},

  ]

  rolecard=[
    {id:1,title:"I'm an owner",content:"Find a trusted local dog lover to take care of your dog when you can't. They'll treat your dog like family."},
    {id:2,title:"I'm a service provider",content:"Earn money by taking care of a pet."},
    {id:3,title:"I'm an Veterinary",content:"Find a large network of clients."},

  ]

}
