import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DialogBoxComponent } from '../dialog-box/dialog-box.component';
import { Animal } from '../modals/Animal';
import { AnimalsService } from '../services/animals.service';

@Component({
  selector: 'app-adopt',
  templateUrl: './adopt.component.html',
  styleUrls: ['./adopt.component.scss']
})
export class AdoptComponent implements OnInit{
  animalsForAdoption:Animal[]=[];
  constructor(private animalsService:AnimalsService,public dialog:MatDialog){}

  openDialog(){
    this.dialog.open(DialogBoxComponent,{
      width:'250px',
      data:"right click"
    })
  }
  ngOnInit(){
    let animals=this.animalsService.getAnimals();
    for(let animal of animals){
      if(animal.forAdoption){
        this.animalsForAdoption.push(animal);
      }
    }
  }
}


