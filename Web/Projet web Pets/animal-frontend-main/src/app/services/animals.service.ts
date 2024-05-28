import { Injectable } from '@angular/core';
import { Animal } from '../modals/Animal';

@Injectable({
  providedIn: 'root'
})
export class AnimalsService {
  animals:Animal[]=[
    {id:1,age:"3 months",forAdoption:true,furColor:"black",img:"../../assets/dog.jpg",petType:"Dog",sex:"male"},
    {id:2,age:"2 months",forAdoption:true,furColor:"white",img:"../../assets/cat.jpg",petType:"Cat",sex:"male"},
    {id:3,age:"1 year",forAdoption:true,furColor:"blue",img:"../../assets/dog.jpg",petType:"Dog",sex:"female"},
    {id:4,age:"2 years",forAdoption:true,furColor:"gray",img:"../../assets/cat.jpg",petType:"Cat",sex:"female"},
  ];
  constructor() { }
  getAnimals(){
    return this.animals;
  }
}
